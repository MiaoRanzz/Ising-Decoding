#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run B30 prerequisites/calibration, then gated formal training on two GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from omegaconf import OmegaConf
from scripts.experiments.qewc_v4.evaluation_common import ARCHITECTURES
from scripts.experiments.qewc_v4.retain_checkpoints import prune

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).parent
PYTHON = Path('/root/miniconda3/envs/ising-decoding/bin/python')
MANIFEST = ROOT / 'experiment/qewc-v4/experiment_manifest.yaml'
AUDIT = ROOT / 'logs/qewc_v4/orchestration/b30'
RESUME = False


def execute(script, arguments, label, roots=(), dry=False, protected_epochs=()):
    if RESUME and (AUDIT / f'{label}.done').is_file():
        print(f'{label}: previously completed, skipping', flush=True)
        return
    if RESUME and script in ('run_continual.py', 'run_oracles.py', 'run_fisher_calibration.py',
                              'run_calibration_grid.py', 'run_formal.py'):
        arguments = [*arguments, '--resume']
    command = [str(PYTHON), '-u', str(SCRIPTS / script), *map(str, arguments)]
    print(label + ': ' + ' '.join(command), flush=True)
    if dry:
        return
    if shutil.disk_usage(ROOT / 'outputs').free < 10 * 1024**3:
        raise RuntimeError('Less than 10 GiB output space; no new job launched')
    with (AUDIT / f'{label}.log').open('a' if RESUME else 'x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                code = None
            for root in roots:
                prune(root, AUDIT / f'{label}.pruned.jsonl', protected_epochs=protected_epochs)
            if code is not None:
                if code:
                    raise subprocess.CalledProcessError(code, command)
                break
    (AUDIT / f'{label}.done').touch()


def task_command(arch, task, name, epochs, samples, gpu):
    spec = ARCHITECTURES[arch]
    cfg = OmegaConf.load(MANIFEST)
    return ['--method', 'seq', '--run-name', name,
            '--base-task', f'T0:{spec.base_config}', '--task', f'{task}:{spec.task_config_dir}/{task}',
            '--order', task, '--expected-stages', '1',
            '--stage0', str(ROOT / cfg.stage0[arch].endpoint.path),
            '--epochs-per-task', str(epochs), '--requested-samples-per-epoch', str(samples),
            '--batch-size', '2048', '--accumulate-steps', '2',
            '--checkpoint-prefix', spec.checkpoint_prefix, '--validation-samples', '65536',
            '--seed', '20260907', '--gpu', str(gpu), '--python', str(PYTHON)]


def prerequisites(arch, gpu, dry):
    smoke = f'qewc_v4/smoke/b30_{arch}'
    execute('run_continual.py', task_command(arch, 'T1', smoke, 1, 8192, gpu),
            f'{arch}_smoke', [ROOT / 'outputs' / smoke], dry)
    if arch == 'htnet_a101':
        for task in ('T1', 'T2', 'T3', 'T4'):
            name = f'qewc_v4/budget_pilot/htnet/{task}_e30'
            execute('run_continual.py', task_command(arch, task, name, 30, 4194304, gpu),
                    f'{arch}_oracle_{task}', [ROOT / 'outputs' / name], dry)
    common = ['--architecture', arch, '--gpu', str(gpu)]
    execute('run_oracles.py', [*common, '--manifest', MANIFEST], f'{arch}_oracles',
            [ROOT / 'outputs/qewc_v4/oracles' / arch], dry)
    execute('run_fisher_calibration.py', [*common, '--manifest', MANIFEST], f'{arch}_fisher', dry=dry)
    calibration = ROOT / 'outputs/qewc_v4/calibration' / arch
    count = 65536 if dry else json.loads((calibration / 'fisher/selection.json').read_text())['selected_fisher_samples']
    execute('calibrate_replay_capacity.py', [*common, '--fisher-state', calibration / f'fisher/T0_n{count}.pt',
            '--output-dir', calibration / 'replay_capacity'], f'{arch}_capacity', dry=dry)
    execute('run_calibration_grid.py', common, f'{arch}_calibration', [calibration], dry)


def formal(arch, gpu, dry):
    prefix = ARCHITECTURES[arch].short
    for suffix in ('T0', 'MIX_TOTAL', 'MIX_EXPOSURE', 'REV_SEQ', 'REV_EWC'):
        run_id = f'{prefix}_{suffix}'
        args = ['--run-id', run_id, '--gpu', str(gpu)]
        execute('run_formal.py', [*args, '--dry-run'], f'{run_id}_dry', dry=dry)
        execute('run_formal.py', args, run_id, [ROOT / 'outputs/qewc_v4/formal' / run_id], dry)


def main():
    global RESUME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    RESUME = args.resume
    assert Path(sys.prefix) == PYTHON.parent.parent
    assert OmegaConf.load(MANIFEST).budget_pilot.selected_B == 30
    assert json.loads((ROOT / 'experiment/qewc-v4/budget_decision.json').read_text())['selected_B'] == 30
    os.environ.update(PYTHONPATH=str(ROOT / 'code'), PYTHON_BIN=str(PYTHON),
                      PREDECODER_PYTHON=str(PYTHON), QEWC_V4_MIN_FREE_GIB='10')
    os.environ['PATH'] = str(PYTHON.parent) + os.pathsep + os.environ['PATH']
    if not args.dry_run:
        AUDIT.mkdir(parents=True, exist_ok=True)
        failure = AUDIT / 'failed.json'
        if RESUME and failure.exists():
            failure.rename(AUDIT / f'failed.previous_{failure.stat().st_mtime_ns}.json')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prerequisites, arch, gpu, args.dry_run)
                       for gpu, arch in enumerate(ARCHITECTURES)]
            for future in futures:
                future.result()
        execute('freeze_selected_parameters.py', [], 'freeze', dry=args.dry_run)
        execute('promote_selected_calibration_runs.py', [], 'promote', dry=args.dry_run)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(formal, arch, gpu, args.dry_run)
                       for gpu, arch in enumerate(ARCHITECTURES)]
            for future in futures:
                future.result()
        if not args.dry_run:
            (AUDIT / 'training.done').touch()
    except Exception as exc:
        if not args.dry_run:
            (AUDIT / 'failed.json').write_text(json.dumps({'error': str(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
