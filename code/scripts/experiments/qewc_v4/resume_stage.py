# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume interrupted stages while retaining auditable loss and Replay state."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

LOSS = re.compile(r'LOSS train ([0-9.eE+-]+) valid ([0-9.eE+-]+)')


def replay_resume_stage(args):
    """Preserve the live snapshot while skipping preceding completed stages."""
    if args.method != 'replay' or not args.resume or args.dry_run:
        return 0
    for stage, task in enumerate(args.order_names, 1):
        models = args.run_root / f'stage_{stage}_{task}' / 'models'
        if (models / f'{args.checkpoint_prefix}.0.{args.epochs_per_task}.pt').exists():
            continue
        if list(models.glob(f'{args.checkpoint_prefix}.0.*.pt')):
            return stage
        return 0
    return 0


def validate_replay(args, task, model, epoch, steps):
    import torch
    state = torch.load(model.with_name(f'checkpoint.0.{epoch}.pt'), map_location='cpu', weights_only=False)
    snapshot = state['metadata']['replay_snapshot']
    buffer = torch.load(args.replay_dir / 'rank_000.pt', map_location='cpu', weights_only=False)
    assert snapshot['task_id'] == task.name
    assert snapshot['epoch'] == buffer['snapshot_epoch'] == epoch
    assert snapshot['global_step'] == buffer['snapshot_global_step'] == epoch * steps
    assert snapshot['world_size'] == buffer['world_size'] == 1 and buffer['rank'] == 0
    assert buffer['capacity'] == buffer['size'] == snapshot['buffer_size_per_rank'] == args.replay_capacity
    for key in ('train_x', 'train_y'):
        assert len(buffer[key]) == buffer['size'] and torch.isfinite(buffer[key]).all()
    assert 'admission_rng_state' in buffer and 'retrieval_rng_state' in buffer
    manifest = json.loads((args.replay_dir / 'balanced_manifest.json').read_text())
    index = args.order_names.index(task.name)
    assert manifest['tasks'] == sorted([args.base_task.name, *args.order_names[:index]])
    assert set(buffer['task_ids']) == set(manifest['tasks'])


def validate_pair(model, epoch, steps):
    import torch
    weights = torch.load(model, map_location='cpu', weights_only=True)
    assert weights and all(torch.isfinite(v).all().item() for v in weights.values())
    state = torch.load(model.with_name(f'checkpoint.0.{epoch}.pt'), map_location='cpu', weights_only=False)
    assert state['epoch'] == epoch and state['global_step'] == epoch * steps
    assert all(k in state for k in ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'))


def combine(old_log, new_log, old_rows, new_rows, start, total, steps):
    matches = list(LOSS.finditer(old_log))
    assert len(matches) >= start > 0
    rows = [r for r in old_rows if r['epoch'] < start] + new_rows
    assert [r['epoch'] for r in rows] == list(range(total))
    assert rows[-1]['optimizer_steps_global'] == total * steps
    text = old_log[:matches[start - 1].end()] + '\n[Resumed from full training checkpoint]\n' + new_log
    assert len(LOSS.findall(text)) == total
    return text, {'epochs': rows}


def prepare(args, task, stage, env, endpoint, repo):
    models = endpoint.parent
    candidates = sorted((int(p.name.split('.')[-2]), p)
                        for p in models.glob(f'{args.checkpoint_prefix}.0.*.pt'))
    if not candidates:
        return None
    epoch, model = candidates[-1]
    if not 0 < epoch < args.epochs_per_task:
        raise RuntimeError('Unexpected partial checkpoint epoch')
    steps = args.steps_per_task // args.epochs_per_task
    validate_pair(model, epoch, steps)
    if args.method == 'replay':
        validate_replay(args, task, model, epoch, steps)
    log_root = repo / 'logs' / args.run_name
    pattern = f'stage_{stage}_{task.name}_*/train.log'
    if stage is None:  # Mix uses run-name_timestamp instead of stage subdirectories.
        pattern = f'{log_root.name}_*/train.log'
        log_root = log_root.parent
    candidates = sorted(p for p in log_root.glob(pattern)
                        if len(LOSS.findall(p.read_text())) >= epoch)
    if not candidates:
        raise RuntimeError('Missing pre-interruption loss history')
    backup = models.parent / 'recovery' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True)
    shutil.copy2(candidates[-1], backup / 'original_train.log')
    shutil.copy2(models.parent / 'resource_metrics.json', backup / 'original_resource_metrics.json')
    if args.method == 'replay':
        shutil.copytree(args.replay_dir, backup / 'replay_snapshot')
    metadata = {'resumed_from_epoch': epoch, 'total_epochs': args.epochs_per_task,
                'checkpoint': str(model), 'original_log': str(candidates[-1]),
                'rng_state_restored': False, 'status': 'running'}
    (backup / 'recovery.json').write_text(json.dumps(metadata, indent=2) + '\n')
    marker = models.parent / 'early_stopping.json'
    if marker.exists():
        shutil.move(str(marker), str(backup / 'original_early_stopping.json'))
    env['FRESH_START'] = '0'
    env['EXTRA_PARAMS'] = '++early_stopping.enabled=False'
    print(f'[QEWC v4 resume] {task.name}: epoch {epoch}, full optimizer/scheduler/scaler state', flush=True)
    return backup, log_root, epoch, steps


def finish(context, args, task, stage, endpoint):
    backup, log_root, epoch, steps = context
    validate_pair(endpoint, args.epochs_per_task, steps)
    prefix = f'stage_{stage}_{task.name}' if stage is not None else Path(args.run_name).name
    newest = sorted(log_root.glob(f'{prefix}_*/train.log'))[-1]
    resource = endpoint.parent.parent / 'resource_metrics.json'
    shutil.copy2(newest, backup / 'resumed_train.log')
    shutil.copy2(resource, backup / 'resumed_resource_metrics.json')
    text, rows = combine((backup / 'original_train.log').read_text(), newest.read_text(),
                        json.loads((backup / 'original_resource_metrics.json').read_text())['epochs'],
                        json.loads(resource.read_text())['epochs'], epoch, args.epochs_per_task, steps)
    combined = log_root / f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_recovered"
    combined.mkdir()
    (combined / 'train.log').write_text(text)
    resource.write_text(json.dumps(rows, indent=2) + '\n')
    metadata = json.loads((backup / 'recovery.json').read_text())
    metadata.update(status='completed', combined_log=str(combined / 'train.log'))
    (backup / 'recovery.json').write_text(json.dumps(metadata, indent=2) + '\n')
