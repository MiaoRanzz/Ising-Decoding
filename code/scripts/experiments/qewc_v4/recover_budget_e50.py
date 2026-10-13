#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover the two September 10 storage-interrupted pilots, then select B.

This deliberately targets only T3/E49 and T4/E46. Original artifacts are
retained. Optimizer/scheduler/scaler state is restored, but the old checkpoints
do not contain RNG state, so continuation is not bitwise reproducible.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from scripts.experiments.qewc_v4 import run_continual as runner
from scripts.experiments.qewc_v4.finish_budget_selection import STAGE0

ROOT = runner.REPO_ROOT
PYTHON = Path('/root/miniconda3/envs/ising-decoding/bin/python')
LOSS = re.compile(r'LOSS train ([0-9.eE+-]+) valid ([0-9.eE+-]+)')


def merge_history(old_log, new_log, old_rows, new_rows, start_epoch):
    """Discard unsaved work from the accepted trajectory, not the archive."""
    matches = list(LOSS.finditer(old_log))
    if len(matches) < start_epoch:
        raise ValueError('Not enough pre-interruption loss rows')
    rows = [row for row in old_rows if row['epoch'] < start_epoch] + new_rows
    if [row['epoch'] for row in rows] != list(range(50)):
        raise ValueError('Recovered resource epochs are not exactly 0..49')
    combined = old_log[:matches[start_epoch - 1].end()] + '\n[Recovery boundary]\n' + new_log
    if len(LOSS.findall(combined)) != 50:
        raise ValueError('Recovered loss history is not exactly 50 rows')
    if rows[-1]['optimizer_steps_global'] != 25600:
        raise ValueError('Recovered optimizer step count is not 25600')
    return combined, {'epochs': rows}


def validate_pair(models, epoch):
    import torch
    weights = torch.load(models / f'HTnet.0.{epoch}.pt', map_location='cpu', weights_only=True)
    if not weights or not all(torch.isfinite(v).all().item() for v in weights.values()):
        raise ValueError('Invalid or nonfinite model state')
    state = torch.load(models / f'checkpoint.0.{epoch}.pt', map_location='cpu', weights_only=False)
    for key in ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'):
        if key not in state:
            raise ValueError(f'Missing {key}')
    if state['epoch'] != epoch or state['global_step'] != epoch * 512:
        raise ValueError('Checkpoint epoch/step mismatch')


def recover(task, epoch, gpu, audit):
    run_name = f'qewc_v4/budget_pilot/htnet/{task}_e50'
    stage = ROOT / 'outputs' / run_name / f'stage_1_{task}'
    models = stage / 'models'
    backup = audit / task
    backup.mkdir()
    log_parent = ROOT / 'logs' / run_name
    old_log = sorted(path for path in log_parent.glob(f'stage_1_{task}_*/train.log')
                     if len(LOSS.findall(path.read_text())) >= epoch)[-1]
    shutil.copy2(old_log, backup / 'original_train.log')
    shutil.copy2(stage / 'resource_metrics.json', backup / 'original_resource_metrics.json')
    validate_pair(models, epoch)
    resume_dir = backup / 'resume_models'
    resume_dir.mkdir()
    for name in (f'HTnet.0.{epoch}.pt', f'checkpoint.0.{epoch}.pt'):
        shutil.copy2(models / name, resume_dir / name)
    endpoint = models / 'HTnet.0.50.pt'
    if endpoint.exists():
        # Preserve the known partial write instead of deleting/overwriting it.
        shutil.move(str(endpoint), str(backup / 'interrupted_HTnet.0.50.pt'))
    argv = [
        '--method', 'seq', '--run-name', run_name,
        '--base-task', 'T0:experiments/qewc_v4/htnet/T0',
        '--task', f'{task}:experiments/qewc_v4/htnet/{task}',
        '--order', task, '--expected-stages', '1', '--stage0', str(STAGE0),
        '--epochs-per-task', '50', '--requested-samples-per-epoch', '4194304',
        '--batch-size', '2048', '--accumulate-steps', '2',
        '--validation-samples', '65536', '--seed', '20260907',
        '--gpu', str(gpu), '--python', str(PYTHON), '--resume',
    ]
    args = runner.parse_args(argv)
    env, _ = runner.stage_environment(args, args.tasks[task], 1, STAGE0)
    env['FRESH_START'] = '0'
    # Public config derives resume_dir from output; overriding it is forbidden.
    # Archiving the partial endpoint leaves the last complete checkpoint pair.
    env['EXTRA_PARAMS'] = ''
    env['PYTHON_BIN'] = str(PYTHON)
    env['PATH'] = str(PYTHON.parent) + os.pathsep + env.get('PATH', '')
    provenance = dict(task=task, resumed_from_epoch=epoch, gpu=gpu,
                      original_log=str(old_log), rng_state_restored=False,
                      status='running', argv=argv)
    (backup / 'recovery.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with (backup / 'launcher.log').open('w') as log:
        subprocess.run(['bash', str(runner.LOCAL_RUN), '9', '9'], env=env,
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    validate_pair(models, 50)
    new_log = sorted(log_parent.glob(f'stage_1_{task}_*/train.log'))[-1]
    if new_log == old_log:
        raise RuntimeError('Missing resumed training log')
    shutil.copy2(new_log, backup / 'resumed_train.log')
    shutil.copy2(stage / 'resource_metrics.json', backup / 'resumed_resource_metrics.json')
    combined_log, resources = merge_history(
        (backup / 'original_train.log').read_text(), new_log.read_text(),
        json.loads((backup / 'original_resource_metrics.json').read_text())['epochs'],
        json.loads((stage / 'resource_metrics.json').read_text())['epochs'], epoch)
    # The budget selector expects one complete trajectory in its latest log.
    combined_dir = log_parent / f"stage_1_{task}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_recovered"
    combined_dir.mkdir()
    (combined_dir / 'train.log').write_text(combined_log)
    (stage / 'resource_metrics.json').write_text(json.dumps(resources, indent=2) + '\n')
    subprocess.run([str(PYTHON), str(Path(runner.__file__)), *argv], cwd=ROOT, check=True)
    provenance.update(status='completed', combined_log=str(combined_dir / 'train.log'))
    (backup / 'recovery.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'{task}: verified E50 and full 50-epoch history', flush=True)


def main():
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        raise RuntimeError(f'Wrong environment: {sys.prefix}')
    # Fail before changing any artifacts if either target filesystem is unwritable.
    for folder in (ROOT / 'outputs/qewc_v4', ROOT / 'logs/qewc_v4'):
        with tempfile.TemporaryFile(dir=folder) as probe:
            probe.write(bytes(16 * 1024 * 1024))
            probe.flush()
            os.fsync(probe.fileno())
    audit = ROOT / 'outputs/qewc_v4/orchestration' / f"e50_recovery_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    audit.mkdir()
    print(f'Recovery audit: {audit}', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(recover, task, epoch, gpu, audit)
                       for task, epoch, gpu in [('T3', 49, 0), ('T4', 46, 1)]]
            for future in futures:
                future.result()
        (audit / 'training.done').touch()
        subprocess.run([str(PYTHON), '-u', str(Path(__file__).with_name('finish_budget_selection.py'))],
                       cwd=ROOT, check=True)
        (audit / 'budget.done').touch()
    except Exception as exc:
        (audit / 'failed.json').write_text(json.dumps({'error': str(exc)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
