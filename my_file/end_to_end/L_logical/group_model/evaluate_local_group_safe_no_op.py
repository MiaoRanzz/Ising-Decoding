#!/usr/bin/env python3
"""Select a group gate and threshold by validation endpoint LER; test once."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from omegaconf import OmegaConf
HERE=Path(__file__).resolve().parent; L_ROOT=HERE.parent; REPO_ROOT=HERE.parents[3]; CODE_ROOT=REPO_ROOT/'code'
for p in (HERE,L_ROOT,CODE_ROOT):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from compare_three_paths import BatchActionModel,baseline_failures,build_matcher,build_model_cfg,load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule,_build_stab_maps
from packet_model.generate_local_risk_dataset import evaluate_actions
from local_group_safe_no_op import GroupGateArchitecture,LocalGroupSafeNoOpGate,split_rows
DEFAULT_SETTINGS=L_ROOT/'end_to_end.yaml'
def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--settings',type=Path,default=DEFAULT_SETTINGS);p.add_argument('--risk-dataset-dir',type=Path);p.add_argument('--checkpoint-dir',type=Path);p.add_argument('--output',type=Path);p.add_argument('--device');return p.parse_args()
def path(v): p=Path(v).expanduser();return p if p.is_absolute() else REPO_ROOT/p
def settings(cli):
    f=cli.settings.resolve();s=OmegaConf.to_container(OmegaConf.load(f).get('group_gate_evaluation',{}),resolve=True)
    if not isinstance(s,dict):raise ValueError('group_gate_evaluation must be a mapping')
    def pick(k,required=False,default=None):
        v=getattr(cli,k,None);v=s.get(k,default) if v is None else v
        if required and v is None:raise ValueError(f'missing group_gate_evaluation.{k} in {f}')
        return v
    return SimpleNamespace(risk_dataset_dir=path(pick('risk_dataset_dir',True)),checkpoint_dir=path(pick('checkpoint_dir',True)),checkpoint_glob=str(pick('checkpoint_glob',default='epoch_*.pt')),output=path(pick('output',True)),batch_size=int(pick('batch_size',True)),train_fraction=float(pick('train_fraction',True)),validation_fraction=float(pick('validation_fraction',True)),split_seed=int(pick('split_seed',True)),num_thresholds=int(pick('num_thresholds',default=21)),device=cli.device if cli.device else pick('device'))
def unpack(packed,rows,shape):
    b=np.unpackbits(packed[rows],axis=1,bitorder='little')[:,:int(np.prod(shape))];return b.reshape(len(rows),*shape).astype(np.uint8)
def summary(x):return {'logical_errors':int(x.sum()),'samples':int(len(x)),'ler':float(x.mean())}
def group_batch(rows,shot_ptr,group_ptr,members):
    gids=[]; desc=[]; ptr=[0]
    for batch,row in enumerate(rows):
        for group in range(int(shot_ptr[row]),int(shot_ptr[row+1])):
            part=np.asarray(members[group_ptr[group]:group_ptr[group+1]],dtype=np.int64);desc.append(np.column_stack((np.full(len(part),batch,dtype=np.int64),part)));ptr.append(ptr[-1]+len(part));gids.append(group)
    return np.asarray(gids,dtype=np.int64),np.concatenate(desc) if desc else np.empty((0,5),np.int64),np.asarray(ptr,dtype=np.int64)
def scores_for(model,rows,x,source,logits,shot_ptr,gptr,members,batch,device):
    score=np.full(int(shot_ptr[-1]),np.nan,dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            rr=rows[start:start+batch];gids,desc,ptr=group_batch(rr,shot_ptr,gptr,members)
            if not len(gids):continue
            action=(logits[rr]>=0).astype(np.float32);feature=np.concatenate((np.array(x[source[rr]],dtype=np.float32,copy=True),np.array(logits[rr],dtype=np.float32,copy=True),action),axis=1)
            out=model(torch.as_tensor(feature,device=device),torch.as_tensor(desc,device=device),torch.as_tensor(ptr,device=device)).cpu().numpy();score[gids]=out
    return score
def apply(actions,rows,scores,threshold,shot_ptr,gptr,members):
    result=actions.copy();accepted=0;active=0
    for local,row in enumerate(rows):
        for group in range(int(shot_ptr[row]),int(shot_ptr[row+1])):
            active+=1
            if scores[group]>=threshold:accepted+=1;continue
            for packet,t,y,x in members[gptr[group]:gptr[group+1]]:
                if packet==0:result[local,0,t,y,x]=result[local,1,t,y,x]=0
                else:result[local,packet+1,t,y,x]=0
    return result,accepted,active
def failures(rows,actions,detsobs,source,num_obs,pipeline,action_model,matcher,device,batch):
    out=[]
    for start in range(0,len(rows),batch):
        rr=rows[start:start+batch];src=source[rr];out.append(evaluate_actions(pipeline,action_model,matcher,np.array(detsobs[src,:-num_obs],dtype=np.uint8,copy=True),np.asarray(detsobs[src,-num_obs:],dtype=np.uint8),actions[start:start+len(rr)],device))
    return np.concatenate(out)
def thresholds(values,n):
    finite=values[np.isfinite(values)]
    return np.unique(np.r_[-np.inf,np.quantile(finite,np.linspace(0,1,n)) if len(finite) else [],np.inf])
def main():
    args=settings(parse_args());meta=json.loads((args.risk_dataset_dir/'metadata.json').read_text());source=np.load(args.risk_dataset_dir/'source_indices.npy',mmap_mode='r');logits=np.load(args.risk_dataset_dir/'proposal_logits.npy',mmap_mode='r');packed=np.load(args.risk_dataset_dir/'proposal_actions_packed.npy',mmap_mode='r');shot_ptr=np.load(args.risk_dataset_dir/'shot_group_ptr.npy',mmap_mode='r');gptr=np.load(args.risk_dataset_dir/'group_member_ptr.npy',mmap_mode='r');members=np.load(args.risk_dataset_dir/'group_members.npy',mmap_mode='r');source_meta,detsobs,x,_=load_corpus(Path(meta['source_dataset_dir']));shape=tuple(meta['proposal_action_shape']);_,val,test=split_rows(len(source),args.train_fraction,args.validation_fraction,args.split_seed);device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'));matcher,num_obs=build_matcher(source_meta);cfg=build_model_cfg(SimpleNamespace(project_config=Path(meta['project_config']),checkpoint=Path(meta['proposal_checkpoint']),model_id=meta['proposal_model_id']),source_meta);maps=_build_stab_maps(int(source_meta['distance']),str(source_meta['code_rotation']));am=BatchActionModel().to(device).eval();pipeline=PreDecoderMemoryEvalModule(am,cfg,maps,device).to(device).eval();val_proposal=unpack(packed,val,shape);test_proposal=unpack(packed,test,shape);val_base=baseline_failures(matcher,np.array(detsobs[source[val]],dtype=np.uint8,copy=True),num_obs,args.batch_size);test_base=baseline_failures(matcher,np.array(detsobs[source[test]],dtype=np.uint8,copy=True),num_obs,args.batch_size);best=None;candidates=sorted(args.checkpoint_dir.glob(args.checkpoint_glob));print(f'[select] {len(candidates)} checkpoints, {len(val)} val, {len(test)} test')
    for ckpt in candidates:
        saved=torch.load(ckpt,map_location=device,weights_only=False);model=LocalGroupSafeNoOpGate(GroupGateArchitecture(**saved['architecture'])).to(device);model.load_state_dict(saved['state_dict']);score=scores_for(model,val,x,source,logits,shot_ptr,gptr,members,args.batch_size,device)
        for th in thresholds(score,args.num_thresholds):
            gated,accepted,active=apply(val_proposal,val,score,th,shot_ptr,gptr,members);fail=failures(val,gated,detsobs,source,num_obs,pipeline,am,matcher,device,args.batch_size);candidate=(int(fail.sum()),str(ckpt),float(th),accepted,active)
            if best is None or candidate<best:best=candidate
    if best is None:raise FileNotFoundError('no gate checkpoint found')
    _,ckpt,th,_,_=best;saved=torch.load(ckpt,map_location=device,weights_only=False);model=LocalGroupSafeNoOpGate(GroupGateArchitecture(**saved['architecture'])).to(device);model.load_state_dict(saved['state_dict']);test_score=scores_for(model,test,x,source,logits,shot_ptr,gptr,members,args.batch_size,device);gated,accepted,active=apply(test_proposal,test,test_score,th,shot_ptr,gptr,members);gate_fail=failures(test,gated,detsobs,source,num_obs,pipeline,am,matcher,device,args.batch_size);proposal_fail=failures(test,test_proposal,detsobs,source,num_obs,pipeline,am,matcher,device,args.batch_size);report={'risk_dataset_dir':str(args.risk_dataset_dir),'selected_checkpoint':ckpt,'gate_threshold':th,'held_out_shots':len(test),'paths':{'pymatching':summary(test_base),'proposal_plus_pymatching':summary(proposal_fail),'local_group_safe_no_op_plus_pymatching':summary(gate_fail)},'group_gate':{'active_groups':active,'accepted_groups':accepted,'accept_coverage':accepted/max(1,active)},'validation_selected_ler':best[0]/len(val)};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(f'[done] report={args.output}')
if __name__=='__main__':main()
