#!/usr/bin/env python3
"""Train a conservative group-level harmful-veto gate."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
HERE=Path(__file__).resolve().parent; L_ROOT=HERE.parent; REPO_ROOT=HERE.parents[3]
for p in (HERE,L_ROOT):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from local_group_safe_no_op import GroupGateArchitecture, LocalGroupSafeNoOpGate, gate_features, split_rows
DEFAULT_SETTINGS=L_ROOT / "end_to_end.yaml"

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--settings",type=Path,default=DEFAULT_SETTINGS); p.add_argument("--risk-dataset-dir",type=Path); p.add_argument("--output-dir",type=Path); p.add_argument("--log-every-batches",type=int); p.add_argument("--device"); return p.parse_args()
def _path(v): p=Path(v).expanduser(); return p if p.is_absolute() else REPO_ROOT/p
def settings(cli):
    file=cli.settings.resolve(); s=OmegaConf.to_container(OmegaConf.load(file).get("group_gate_training",{}),resolve=True)
    if not isinstance(s,dict): raise ValueError("group_gate_training must be a mapping")
    def pick(k,required=False,default=None):
        v=getattr(cli,k,None); v=s.get(k,default) if v is None else v
        if required and v is None: raise ValueError(f"missing group_gate_training.{k} in {file}")
        return v
    return SimpleNamespace(risk_dataset_dir=_path(pick("risk_dataset_dir",True)),output_dir=_path(pick("output_dir",True)),batch_size=int(pick("batch_size",True)),epochs=int(pick("epochs",True)),learning_rate=float(pick("learning_rate",True)),min_learning_rate=float(pick("min_learning_rate",default=0)),weight_decay=float(pick("weight_decay",default=0)),lr_scheduler=str(pick("lr_scheduler",default="cosine")).lower(),train_fraction=float(pick("train_fraction",default=.7)),validation_fraction=float(pick("validation_fraction",default=.15)),split_seed=int(pick("split_seed",default=20260808)),neutral_per_informative=float(pick("neutral_per_informative",default=10)),max_positive_weight=float(pick("max_positive_weight",default=100)),num_workers=int(pick("num_workers",default=0)),log_every_batches=int(pick("log_every_batches",default=256)),device=cli.device if cli.device else pick("device"),architecture=GroupGateArchitecture(**s.get("architecture",{})))

class GroupDataset(Dataset):
    def __init__(self,risk, rows):
        self.meta=json.loads((risk/"metadata.json").read_text());
        if self.meta.get("artifact")!="local_group_safe_no_op_risk_dataset": raise ValueError("not a group risk dataset")
        self.rows=np.asarray(rows,dtype=np.int64); source=Path(self.meta["source_dataset_dir"]); self.x=np.load(source/"train_x.npy",mmap_mode="r"); self.indices=np.load(risk/"source_indices.npy",mmap_mode="r"); self.logits=np.load(risk/"proposal_logits.npy",mmap_mode="r"); self.ptr=np.load(risk/"shot_group_ptr.npy",mmap_mode="r"); self.gptr=np.load(risk/"group_member_ptr.npy",mmap_mode="r"); self.members=np.load(risk/"group_members.npy",mmap_mode="r"); self.effect=np.load(risk/"group_effect.npy",mmap_mode="r")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=int(self.rows[i]); source=int(self.indices[row]); action=(self.logits[row]>=0).astype(np.float32); features=np.concatenate((self.x[source],self.logits[row].astype(np.float32),action),axis=0)
        a,b=int(self.ptr[row]),int(self.ptr[row+1]); ma,mb=int(self.gptr[a]),int(self.gptr[b])
        ptr=self.gptr[a:b+1]-ma; return features, np.asarray(ptr,dtype=np.int64),np.asarray(self.members[ma:mb],dtype=np.int64),np.asarray(self.effect[a:b],dtype=np.int8)

def collate(items):
    features=[]; members=[]; ptr=[0]; effects=[]
    for batch,(x,local_ptr,local_members,effect) in enumerate(items):
        features.append(torch.from_numpy(np.array(x,copy=True))); m=np.array(local_members,copy=True); m=np.column_stack((np.full(len(m),batch,dtype=np.int64),m)); members.append(m); effects.append(effect)
        # ``local_ptr`` is already cumulative within this shot.  Convert it
        # back to one group's member count before extending the batch-global
        # CSR pointer; adding every cumulative endpoint would over-count.
        previous = int(local_ptr[0])
        for end in local_ptr[1:]:
            ptr.append(ptr[-1] + int(end) - previous)
            previous = int(end)
    return torch.stack(features),torch.as_tensor(np.concatenate(members) if members else np.empty((0,5),np.int64)),torch.as_tensor(ptr),torch.as_tensor(np.concatenate(effects))

def group_labels(dataset,rows):
    out=[]
    for row in rows: out.append(np.asarray(dataset.effect[int(dataset.ptr[row]):int(dataset.ptr[row+1])]))
    return np.concatenate(out)

def main():
    args=settings(parse_args()); meta=json.loads((args.risk_dataset_dir/"metadata.json").read_text()); total=int(meta["num_samples"]); train_rows,val_rows,_=split_rows(total,args.train_fraction,args.validation_fraction,args.split_seed)
    probe=GroupDataset(args.risk_dataset_dir,np.empty(0,dtype=np.int64)); train_effect=group_labels(probe,train_rows); informative=(train_effect!=0); neutral=train_effect==0; keep=min(1.,args.neutral_per_informative*int(informative.sum())/max(1,int(neutral.sum())))
    # Retain all informative shots; resample neutral-only context each epoch.
    informative_shots = np.empty(len(train_rows), dtype=bool)
    for index, row in enumerate(train_rows):
        effect=np.asarray(probe.effect[int(probe.ptr[row]):int(probe.ptr[row+1])])
        informative_shots[index] = np.any(effect != 0)
    val=GroupDataset(args.risk_dataset_dir,val_rows)
    val_loader=DataLoader(val,args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); model=LocalGroupSafeNoOpGate(args.architecture).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.min_learning_rate) if args.lr_scheduler=="cosine" else None
    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/"settings.json").write_text(json.dumps({"architecture":args.architecture.to_dict(),"risk_dataset_dir":str(args.risk_dataset_dir),"split_seed":args.split_seed,"train_fraction":args.train_fraction,"validation_fraction":args.validation_fraction,"neutral_keep_probability":keep,"neutral_resampling":"per_epoch","gate_target":"harmful_veto"},indent=2))
    print(
        f"[data] candidate_train_shots={len(train_rows)}, informative_train_shots={int(informative_shots.sum())}, "
        f"val_shots={len(val)}, candidate_groups(helpful/neutral/harmful)="
        f"{(train_effect==1).sum()}/{(train_effect==0).sum()}/{(train_effect==-1).sum()}, "
        f"neutral_keep={keep:.6f}; neutral-only shots are resampled every epoch"
    )
    training_started = time.time()
    for epoch in range(1,args.epochs+1):
        epoch_rng=np.random.default_rng(args.split_seed + epoch); selected=informative_shots | (epoch_rng.random(len(train_rows)) < keep); train_rows_epoch=train_rows[selected]
        train=GroupDataset(args.risk_dataset_dir,train_rows_epoch); train_loader=DataLoader(train,args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate)
        sampled_groups=int(np.asarray(probe.ptr[train_rows_epoch + 1] - probe.ptr[train_rows_epoch]).sum())
        print(f"[epoch {epoch:03d}] sampled_train_shots={len(train)}, sampled_groups={sampled_groups}")
        model.train(); started=time.time(); losses=[]
        for step,(x,members,ptr,effect) in enumerate(train_loader,1):
            x,members,ptr,effect=x.to(device),members.to(device),ptr.to(device),effect.to(device); logits=model(x,members,ptr); target=(effect==-1).float(); weight=torch.where(effect==0,torch.full_like(target,keep),torch.ones_like(target)); pos=min(args.max_positive_weight,(weight.sum()-weight[target.bool()].sum()).item()/max(1,target.sum().item())); loss=F.binary_cross_entropy_with_logits(logits,target,weight=weight,pos_weight=torch.tensor(pos,device=device)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            if step%args.log_every_batches==0 or step==len(train_loader):
                epoch_elapsed = time.time() - started
                epoch_eta = epoch_elapsed / step * (len(train_loader) - step)
                total_elapsed = time.time() - training_started
                completed_batches = (epoch - 1) * len(train_loader) + step
                total_batches_estimate = args.epochs * len(train_loader)
                total_eta = total_elapsed / completed_batches * max(0, total_batches_estimate - completed_batches)
                print(
                    f"[epoch {epoch:03d}] batch {step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-args.log_every_batches:]):.5f} "
                    f"lr={opt.param_groups[0]['lr']:.3g} "
                    f"epoch_eta={epoch_eta:.0f}s total_eta~{total_eta:.0f}s"
                )
        model.eval(); vl=[]
        with torch.no_grad():
            for x,members,ptr,effect in val_loader:
                logits=model(x.to(device),members.to(device),ptr.to(device)); vl.append(F.binary_cross_entropy_with_logits(logits,(effect.to(device)==-1).float()).item())
        torch.save({"epoch":epoch,"state_dict":model.state_dict(),"architecture":args.architecture.to_dict(),"gate_target":"harmful_veto"},args.output_dir/f"epoch_{epoch:03d}.pt"); print(f"[epoch {epoch:03d}] train_loss={np.mean(losses):.5f} val_harmful_bce={np.mean(vl):.5f} lr={opt.param_groups[0]['lr']:.3g}")
        if sched: sched.step()
if __name__=="__main__": main()
