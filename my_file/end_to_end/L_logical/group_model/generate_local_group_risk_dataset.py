#!/usr/bin/env python3
"""Generate group-level counterfactual labels for the local safe-no-op gate."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
L_ROOT, REPO_ROOT, CODE_ROOT = HERE.parent, HERE.parents[3], HERE.parents[3] / "code"
for path in (HERE, L_ROOT, CODE_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from compare_three_paths import BatchActionModel, _load_model, build_matcher, build_model_cfg, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from training.precision import match_input_to_model_memory_format
from packet_model.generate_local_risk_dataset import evaluate_actions
from local_group_safe_no_op import GroupingConfig, build_groups, proposal_actions_from_logits

DEFAULT_SETTINGS = L_ROOT / "end_to_end.yaml"

def parse_args():
    p = argparse.ArgumentParser(description="Generate fixed-group safe-no-op counterfactual labels.")
    p.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    p.add_argument("--source-dataset-dir", type=Path); p.add_argument("--project-config", type=Path)
    p.add_argument("--checkpoint", type=Path); p.add_argument("--model-id", type=int)
    p.add_argument("--output-dir", type=Path); p.add_argument("--num-samples", type=int)
    p.add_argument("--proposal-batch-size", type=int); p.add_argument("--counterfactual-batch-size", type=int)
    p.add_argument("--seed", type=int); p.add_argument("--device")
    return p.parse_args()

def _path(value):
    path = Path(value).expanduser(); return path if path.is_absolute() else REPO_ROOT / path

def settings(cli):
    file = cli.settings.expanduser().resolve(); section = OmegaConf.to_container(OmegaConf.load(file).get("group_risk_generation", {}), resolve=True)
    if not isinstance(section, dict): raise ValueError("group_risk_generation must be a mapping")
    def pick(name, required=False, fallback=None):
        value = getattr(cli, name, None)
        if value is None: value = section.get(name, fallback)
        if required and value is None: raise ValueError(f"missing group_risk_generation.{name} in {file}")
        return value
    grouping = GroupingConfig(**section.get("grouping", {}))
    return SimpleNamespace(source_dataset_dir=_path(pick("source_dataset_dir", True)), project_config=_path(pick("project_config", True)), checkpoint=_path(pick("checkpoint", True)), model_id=pick("model_id"), output_dir=_path(pick("output_dir", True)), num_samples=pick("num_samples"), proposal_batch_size=int(pick("proposal_batch_size", True)), counterfactual_batch_size=int(pick("counterfactual_batch_size", True)), seed=int(pick("seed", fallback=20260808)), device=pick("device"), grouping=grouping)

def unpack_one(packed, shape):
    bits = np.unpackbits(packed, bitorder="little")[:int(np.prod(shape))]
    return bits.reshape(shape).astype(np.uint8, copy=False)

def remove_group(actions, members):
    for packet, t, y, x in members:
        if packet == 0: actions[0, t, y, x] = actions[1, t, y, x] = 0
        else: actions[packet + 1, t, y, x] = 0

def build_layout(actions_packed, shape, output, grouping):
    shots = actions_packed.shape[0]; counts = np.zeros(shots + 1, dtype=np.int64); member_counts = np.zeros(shots, dtype=np.int64)
    for shot in range(shots):
        groups = build_groups(unpack_one(actions_packed[shot], shape), grouping)
        counts[shot + 1], member_counts[shot] = len(groups), sum(len(g) for g in groups)
    shot_ptr = np.cumsum(counts); total_groups, total_members = int(shot_ptr[-1]), int(member_counts.sum())
    group_ptr = np.lib.format.open_memmap(output / "group_member_ptr.npy", mode="w+", dtype=np.int64, shape=(total_groups + 1,))
    members = np.lib.format.open_memmap(output / "group_members.npy", mode="w+", dtype=np.int16, shape=(total_members, 4))
    cursor = 0; group_ptr[0] = 0
    for shot in range(shots):
        groups = build_groups(unpack_one(actions_packed[shot], shape), grouping)
        base = int(shot_ptr[shot])
        for offset, group in enumerate(groups):
            end = cursor + len(group); members[cursor:end] = group; cursor = end; group_ptr[base + offset + 1] = cursor
    np.save(output / "shot_group_ptr.npy", shot_ptr)
    del group_ptr, members
    return total_groups, total_members

def main():
    args = settings(parse_args())
    if not args.checkpoint.is_file(): raise FileNotFoundError(f"proposal checkpoint not found: {args.checkpoint}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    meta, dets_obs, train_x, _ = load_corpus(args.source_dataset_dir); total = int(meta["num_samples"])
    requested = total if args.num_samples is None else int(args.num_samples)
    if not 0 < requested <= total: raise ValueError(f"num_samples must be in [1, {total}]")
    rng=np.random.default_rng(args.seed); source_indices=np.arange(total,dtype=np.int64) if requested==total else np.sort(rng.choice(total, requested, replace=False))
    args.output_dir.mkdir(parents=True); np.save(args.output_dir / "source_indices.npy", source_indices)
    matcher, num_obs = build_matcher(meta)
    if num_obs != 1: raise NotImplementedError("currently supports one logical observable")
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); cfg=build_model_cfg(SimpleNamespace(project_config=args.project_config, checkpoint=args.checkpoint, model_id=args.model_id), meta)
    print(f"[load] proposal={args.checkpoint}, device={device}")
    model=_load_model(cfg, SimpleNamespace(rank=0,world_size=1,device=device)).eval(); maps=_build_stab_maps(int(meta["distance"]),str(meta["code_rotation"]))
    action_model=BatchActionModel().to(device).eval(); pipeline=PreDecoderMemoryEvalModule(action_model,cfg,maps,device).to(device).eval()
    shape=(4,int(meta["n_rounds"]),int(meta["distance"]),int(meta["distance"])); logits=np.lib.format.open_memmap(args.output_dir / "proposal_logits.npy",mode="w+",dtype=np.float16,shape=(requested,*shape)); packed=np.lib.format.open_memmap(args.output_dir / "proposal_actions_packed.npy",mode="w+",dtype=np.uint8,shape=(requested,(int(np.prod(shape))+7)//8))
    with torch.no_grad():
        for start in range(0,requested,args.proposal_batch_size):
            end=min(start+args.proposal_batch_size,requested); x=torch.as_tensor(np.array(train_x[source_indices[start:end]],dtype=np.float32,copy=True),device=device)
            output=model(match_input_to_model_memory_format(x,model)); action=proposal_actions_from_logits(output).to(torch.uint8).cpu().numpy()
            logits[start:end]=output.float().cpu().numpy().astype(np.float16); packed[start:end]=np.packbits(action.reshape(end-start,-1),axis=1,bitorder="little")
            print(f"[proposal] {end}/{requested} shots")
    del logits
    groups, members_count=build_layout(packed,shape,args.output_dir,args.grouping); print(f"[grouping] {groups} groups, {members_count} packet members")
    shot_ptr=np.load(args.output_dir / "shot_group_ptr.npy",mmap_mode="r"); group_ptr=np.load(args.output_dir / "group_member_ptr.npy",mmap_mode="r"); members=np.load(args.output_dir / "group_members.npy",mmap_mode="r"); effects=np.lib.format.open_memmap(args.output_dir / "group_effect.npy",mode="w+",dtype=np.int8,shape=(groups,))
    for start in range(0,requested,args.proposal_batch_size):
        end=min(start+args.proposal_batch_size,requested); base_actions=np.stack([unpack_one(packed[i],shape) for i in range(start,end)]); source=source_indices[start:end]; dets=np.array(dets_obs[source,:-num_obs],dtype=np.uint8,copy=True); obs=np.asarray(dets_obs[source,-num_obs:],dtype=np.uint8)
        base=evaluate_actions(pipeline,action_model,matcher,dets,obs,base_actions,device); candidate=[]
        for local in range(end-start): candidate += [(local,g) for g in range(int(shot_ptr[start+local]),int(shot_ptr[start+local+1]))]
        for off in range(0,len(candidate),args.counterfactual_batch_size):
            batch=candidate[off:off+args.counterfactual_batch_size]; rows=np.asarray([v[0] for v in batch]); cf=np.array(base_actions[rows],copy=True)
            for i,(_,group) in enumerate(batch): remove_group(cf[i],members[group_ptr[group]:group_ptr[group+1]])
            failure=evaluate_actions(pipeline,action_model,matcher,dets[rows],obs[rows],cf,device)
            effects[[g for _,g in batch]]=failure.astype(np.int8)-base[rows].astype(np.int8)
        print(f"[counterfactual] {end}/{requested} shots")
    labels={"helpful":int((effects==1).sum()),"neutral":int((effects==0).sum()),"harmful":int((effects==-1).sum())}; del effects
    payload={"schema_version":1,"artifact":"local_group_safe_no_op_risk_dataset","source_dataset_dir":str(args.source_dataset_dir.resolve()),"project_config":str(args.project_config.resolve()),"source_num_samples":total,"num_samples":requested,"distance":int(meta["distance"]),"n_rounds":int(meta["n_rounds"]),"basis":str(meta["basis"]),"code_rotation":str(meta["code_rotation"]),"proposal_checkpoint":str(args.checkpoint.resolve()),"proposal_model_id":int(cfg.model_id),"grouping":args.grouping.__dict__,"effect_definition":"failure_without_full_group - failure_with_full_proposal","labels":labels,"files":{"source_indices":"source_indices.npy","proposal_logits":"proposal_logits.npy","proposal_actions_packed":"proposal_actions_packed.npy","shot_group_ptr":"shot_group_ptr.npy","group_member_ptr":"group_member_ptr.npy","group_members":"group_members.npy","group_effect":"group_effect.npy"},"proposal_action_shape":list(shape),"proposal_action_pack_bitorder":"little"}
    (args.output_dir / "metadata.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(f"[done] wrote group-risk dataset to {args.output_dir}")
if __name__ == "__main__": main()
