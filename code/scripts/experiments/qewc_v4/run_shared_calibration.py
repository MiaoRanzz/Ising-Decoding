#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTNet-only two-GPU selection; transfer hyperparameters, not Fisher tensors."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from scripts.experiments.qewc_v4 import run_b30_pipeline as pipeline
from scripts.experiments.qewc_v4.evaluation_common import run_ids

ROOT = pipeline.ROOT
PYTHON = pipeline.PYTHON
CAL = ROOT / 'outputs/qewc_v4/calibration'


def wait_for_calibration(pid):
    summary = CAL / 'htnet_a101/calibration_summary.json'
    print(f'Waiting for existing HTNet calibration PID={pid}; no duplicate launch', flush=True)
    while True:
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()
            if stat[0] == 'Z':
                break
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError:
            break
        if 'run_calibration_grid.py' not in command or '--architecture htnet_a101' not in command:
            raise RuntimeError('Calibration PID identity mismatch')
        time.sleep(30)
    if not summary.is_file():
        raise RuntimeError('Calibration exited without summary; downstream training blocked')
    result = json.loads(summary.read_text())
    if result.get('architecture') != 'htnet_a101' or result.get('budget_B') != 30:
        raise RuntimeError('Calibration summary architecture/budget mismatch')
    for key in ('ewc_selection', 'replay_selection'):
        selection = json.loads(Path(result[key]).read_text())
        if not selection.get('selected'):
            raise RuntimeError(f'Incomplete selection: {key}')
    (pipeline.AUDIT / 'htnet_dualgpu.done').touch()


def transferred_summary(source, fisher, capacity):
    result = dict(source)
    result.update(architecture='ising_fast', parameter_source_architecture='htnet_a101',
                  independent_parameter_search=False,
                  initial_fisher_path=str(fisher.relative_to(ROOT)),
                  initial_fisher_sha256=hashlib.sha256(fisher.read_bytes()).hexdigest(),
                  ewc_state_bytes=fisher.stat().st_size,
                  replay_capacity_samples=int(capacity['capacity_samples']),
                  replay_target_bytes=int(capacity['target_bytes']),
                  replay_byte_tolerance=float(capacity['byte_tolerance']),
                  ewc_candidates_evaluated=[], replay_candidates_evaluated=[],
                  source_summary=str(CAL / 'htnet_a101/calibration_summary.json'),
                  source_summary_sha256=hashlib.sha256((CAL / 'htnet_a101/calibration_summary.json').read_bytes()).hexdigest())
    return result


def formal(architecture, gpu):
    for run_id in run_ids(architecture):
        root = ROOT / 'outputs/qewc_v4/formal' / run_id
        if (root / 'run_manifest.json').is_file():
            record = json.loads((root / 'run_manifest.json').read_text())
            assert record.get('legacy_epochs_per_task', record.get('legacy_epochs_per_task_reference')) == 30
            for value in record.get('endpoints', record.get('checkpoints', {})).values():
                assert hashlib.sha256(Path(value['path']).read_bytes()).hexdigest() == value['sha256']
            print(f'{run_id}: verified promoted/completed training', flush=True)
            continue
        args = ['--run-id', run_id, '--gpu', str(gpu), '--manifest',
                str(ROOT / 'outputs/qewc_v4/frozen_experiment_manifest.yaml')]
        pipeline.execute('run_formal.py', [*args, '--dry-run'], f'{run_id}_dry')
        protected = tuple(range(30, 151, 30)) if 'MIX' in run_id else ()
        pipeline.execute('run_formal.py', args, run_id, [root], protected_epochs=protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--adopt-pid', type=int)
    mode.add_argument('--wait-calibration-pid', type=int)
    args = parser.parse_args()
    assert Path(sys.prefix) == PYTHON.parent.parent
    pipeline.AUDIT = ROOT / 'logs/qewc_v4/orchestration/shared_htnet'
    pipeline.AUDIT.mkdir(parents=True, exist_ok=True)
    lock = (pipeline.AUDIT / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipeline.RESUME = True
    os.environ.update(PYTHONPATH=str(ROOT / 'code'), PYTHON_BIN=str(PYTHON),
                      PREDECODER_PYTHON=str(PYTHON), QEWC_V4_MIN_FREE_GIB='10')
    os.environ['PATH'] = str(PYTHON.parent) + os.pathsep + os.environ['PATH']
    try:
        command = ['--architecture', 'htnet_a101', '--gpus', '0,1']
        if args.adopt_pid:
            command += ['--adopt-pid', str(args.adopt_pid)]
        if args.wait_calibration_pid:
            wait_for_calibration(args.wait_calibration_pid)
        else:
            pipeline.execute('run_calibration_grid.py', command, 'htnet_dualgpu')
        source = json.loads((CAL / 'htnet_a101/calibration_summary.json').read_text())
        assert source['budget_B'] == 30
        count = int(source['fisher_samples'])
        fisher = CAL / f'ising_fast/fisher/T0_n{count}.pt'
        if not fisher.is_file():
            raise RuntimeError(f'Missing architecture-specific Fisher: {fisher}')
        # This only matches storage bytes; it does not select a Replay ratio.
        capacity_dir = CAL / 'ising_fast/replay_capacity_htnet_transfer'
        pipeline.execute('calibrate_replay_capacity.py', ['--architecture', 'ising_fast',
                         '--gpu', '1', '--fisher-state', str(fisher), '--output-dir', str(capacity_dir)],
                         'ising_transferred_capacity')
        capacity = json.loads((capacity_dir / 'capacity_selection.json').read_text())
        summary = transferred_summary(source, fisher, capacity)
        output = CAL / 'ising_fast/calibration_summary.json'
        if output.exists() and json.loads(output.read_text()).get('parameter_source_architecture') != 'htnet_a101':
            raise RuntimeError('Refusing to overwrite an independent calibration summary')
        output.write_text(json.dumps(summary, indent=2) + '\n')
        pipeline.execute('freeze_selected_parameters.py', [], 'freeze')
        pipeline.execute('promote_selected_calibration_runs.py', [], 'promote')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(formal, arch, gpu)
                       for gpu, arch in enumerate(('htnet_a101', 'ising_fast'))]
            for future in futures:
                future.result()
        (pipeline.AUDIT / 'training.done').touch()
    except Exception as exc:
        (pipeline.AUDIT / 'failed.json').write_text(json.dumps({'error': str(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
