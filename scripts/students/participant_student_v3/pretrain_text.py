from __future__ import annotations
import argparse,json,math,pickle,random,re,urllib.request,shutil,subprocess,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.utils.class_weight import compute_class_weight

def one(root,pattern):
    h=list(Path(root).rglob(pattern)); assert len(h)==1,(pattern,len(h)); return h[0]

def read_split(root,name,exclude=None):
    d=pd.read_csv(one(root,name)); c={x.lower():x for x in d.columns}
    z=pd.DataFrame({"participant_id":d[c["participant_id"]].astype(int),"label":d[c["phq8_binary"]].astype(int)})
    if exclude: z=z.loc[~z.participant_id.isin(set(exclude))]
    return z.sort_values("participant_id").reset_index(drop=True)

def transcript_map(root,pids):
    want=set(map(int,pids)); out={}
    for p in Path(root).rglob("*_TRANSCRIPT.csv"):
        m=re.fullmatch(r"(\d+)_TRANSCRIPT\.csv",p.name,re.I)
        if m and int(m.group(1)) in want: out.setdefault(int(m.group(1)),[]).append(p)
    for pid in want: assert len(out.get(pid,[]))==1,f"{pid}: transcript coverage={len(out.get(pid,[]))}"
    return {k:v[0] for k,v in out.items()}

def participant_doc(path):
    d=pd.read_csv(path,sep="\t"); d.columns=d.columns.str.strip().str.lower()
    if not {"speaker","value"}<=set(d.columns):
        d=pd.read_csv(path,sep=None,engine="python"); d.columns=d.columns.str.strip().str.lower()
    assert {"speaker","value"}<=set(d.columns),path
    v=d.loc[d.speaker.astype(str).str.strip().str.lower().eq("participant"),"value"].fillna("").astype(str)
    # Idiap data README specifies concatenation of all utterances from the selected speaker.
    return " ".join(v.str.replace(r"\s+"," ",regex=True).str.strip().tolist()).strip()

def published_params(src):
    import optuna
    rel=Path("output/Participant/21_induct-gcn[original-features250]/db.sqlite3")
    source_db=Path(src)/rel
    runtime_db=Path("/content/idiap_participant_original_features250_optuna_runtime.db")
    public_url=("https://raw.githubusercontent.com/idiap/bias_in_daic-woz/main/"
                "output/Participant/21_induct-gcn%5Boriginal-features250%5D/db.sqlite3")

    # Always work on a disposable /content copy. Never mutate the user's
    # pinned Idiap checkout in Drive.
    if source_db.exists():
        if (not runtime_db.exists()
            or runtime_db.stat().st_size != source_db.stat().st_size
            or runtime_db.stat().st_mtime < source_db.stat().st_mtime):
            shutil.copy2(source_db,runtime_db)
        source=str(source_db)
    else:
        if not runtime_db.exists() or runtime_db.stat().st_size < 100000:
            print("Published Idiap Optuna DB missing from cached clone; fetching public experiment DB...")
            urllib.request.urlretrieve(public_url,runtime_db)
        source=public_url

    if not runtime_db.exists() or runtime_db.stat().st_size < 100000:
        raise FileNotFoundError(f"Could not obtain published Idiap Optuna database: {runtime_db}")

    storage=f"sqlite:///{runtime_db}"

    def summaries():
        return optuna.study.get_all_study_summaries(storage)

    try:
        ss=summaries()
    except RuntimeError as e:
        if "table schema" not in str(e).lower() and "not compatible" not in str(e).lower():
            raise
        print("Upgrading disposable published Optuna DB schema for current runtime...")
        optuna_cli=shutil.which("optuna")
        if not optuna_cli:
            raise RuntimeError(
                "Optuna package is importable but the 'optuna' CLI executable is not on PATH. "
                "Run: pip install -U 'optuna>=4,<5'"
            )
        subprocess.run(
            [optuna_cli,"storage","upgrade","--storage",storage],
            check=True
        )
        ss=summaries()

    assert len(ss)>=1,"No study in published Idiap database"
    studies=[optuna.load_study(study_name=x.study_name,storage=storage) for x in ss]
    completed=[q for q in studies if len(q.trials)>0]
    assert completed,"Published Idiap database has no trials"
    study=max(completed,key=lambda q: float(q.best_value))
    p=study.best_trial.params
    assert "learning_rate" in p and "num_steps" in p,p
    return {
        "study_name":study.study_name,
        "best_value":float(study.best_value),
        "learning_rate":float(p["learning_rate"]),
        "num_steps":int(p["num_steps"]),
        "database_path":str(runtime_db),
        "database_source":source,
        "runtime_optuna_version":optuna.__version__,
        "source_database_never_modified":True,
    }

def select_vectorizer(docs,y,k=250):
    base=TfidfVectorizer(stop_words="english"); X=base.fit_transform(docs)
    terms=base.get_feature_names_out(); sel=SelectKBest(f_classif,k=min(k,X.shape[1])).fit(X,y)
    vocab=terms[sel.get_support()].tolist(); assert len(vocab)==k,len(vocab)
    v=TfidfVectorizer(stop_words="english",vocabulary=vocab); v.fit(docs); return v

def build_graph(docs,v,window=3):
    X=torch.tensor(v.transform(docs).toarray(),dtype=torch.float32)
    V=len(v.vocabulary_); tok=v.build_analyzer(); total=0
    freq=defaultdict(int); co=defaultdict(lambda:defaultdict(int))
    for doc in docs:
        w=[x for x in tok(doc) if x in v.vocabulary_]; finish=False; i=0
        while not finish:
            win=set(w[i:i+window])
            for a in win:
                freq[a]+=1
                for b in win:
                    if a!=b: co[a][b]+=1
            finish=i+window>=len(w); total+=1; i+=1
    A=torch.zeros((V+len(docs),V+len(docs)),dtype=torch.float32); idx=v.vocabulary_
    for a,fa in freq.items():
        ia=idx[a]; A[ia,ia]=fa
        for b,c in co[a].items():
            if b in idx: A[ia,idx[b]]=c
    invalid=A[:V,:V].eq(0)
    W=A[:V,:V].clone(); Wlog=torch.zeros_like(W); mask=W.gt(0); Wlog[mask]=torch.log(W[mask])
    diag=torch.diag(Wlog).clone()
    Wlog=Wlog-diag.view(-1,1)-diag.view(1,-1)+math.log(max(total,1))
    Wlog[invalid]=0; Wlog[Wlog<0]=0; A[:V,:V]=Wlog
    A[:V,V:]=X.T; A[range(V+len(docs)),range(V+len(docs))]=1
    deg=A.sum(1).clamp_min(1e-12); inv=deg.rsqrt(); A=inv[:,None]*A*inv[None,:]
    H0=torch.cat([torch.eye(V),X],dim=0); conv0=A@H0
    return A,conv0,X

class InductText(nn.Module):
    def __init__(self,V,d=64,drop=.5):
        super().__init__(); self.V=int(V); self.H_1_words=None
        self.h1=nn.Sequential(nn.Linear(V,d,bias=False),nn.ReLU(),nn.Dropout(drop))
        self.out=nn.Linear(d,2,bias=False)

    def train_repr_logits(self,A,conv0):
        # Match author cross_entropy_loss_on_document_nodes(): H_1_words is
        # captured from the dropout-active TRAIN forward pass and then reused
        # during DEV inference.
        H=self.h1(conv0)
        self.H_1_words=H[:self.V].detach()
        R=A@H
        return R,self.out(R)

    def dev_repr_logits(self,Xdev,Hwords=None):
        Hwords=self.H_1_words if Hwords is None else Hwords
        assert Hwords is not None
        Hwords=Hwords.to(Xdev.device)
        n,V=Xdev.shape
        H0=torch.cat([torch.eye(V,device=Xdev.device),Xdev],dim=0)
        B=torch.zeros((n,V+n),device=Xdev.device); B[:,:V]=Xdev; B[:,V:]=torch.eye(n,device=Xdev.device)
        Hdev=self.h1(B@H0)
        Hall=torch.cat([Hwords[:V],Hdev],dim=0)
        R=B@Hall
        return R,self.out(R)

def metrics(y,p):
    q=p[:,1]; pred=p.argmax(1)
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro")),
            "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
            "auroc":float(roc_auc_score(y,q)),"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--daic-root",required=True); ap.add_argument("--idiap-source",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,default=17); ap.add_argument("--eval-every",type=int,default=1)
    a=ap.parse_args(argv); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    root,src,out=Path(a.daic_root),Path(a.idiap_source),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tr=read_split(root,"train_split_Depression_AVEC2017.csv"); dv=read_split(root,"dev_split_Depression_AVEC2017.csv",[440])
    assert len(tr)==107 and len(dv)==34 and not set(tr.participant_id)&set(dv.participant_id)
    tm=transcript_map(root,pd.concat([tr.participant_id,dv.participant_id]))
    tr_docs=[participant_doc(tm[int(x)]) for x in tr.participant_id]; dv_docs=[participant_doc(tm[int(x)]) for x in dv.participant_id]
    assert all(tr_docs) and all(dv_docs)
    ytr=tr.label.to_numpy(int); ydv=dv.label.to_numpy(int)
    v=select_vectorizer(tr_docs,ytr,250); A,conv0,_=build_graph(tr_docs,v,3); Xdv=torch.tensor(v.transform(dv_docs).toarray(),dtype=torch.float32)
    pub=published_params(src); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A,conv0,Xdv=A.to(device),conv0.to(device),Xdv.to(device); yy=torch.tensor(ytr,dtype=torch.long,device=device)
    model=InductText(250).to(device); opt=torch.optim.AdamW(model.parameters(),lr=pub["learning_rate"])
    cw=torch.tensor(compute_class_weight(class_weight="balanced",classes=np.array([0,1]),y=ytr),dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=cw); best=(-1.,-1.,-1.); best_epoch=0; ckpt=out/"best.pt"; hist=[]
    print("TEXT v3 InducT-style | params:",sum(p.numel() for p in model.parameters()),"| published study:",pub)
    for epoch in range(1,pub["num_steps"]+1):
        model.train(); opt.zero_grad(set_to_none=True); R,z=model.train_repr_logits(A,conv0); loss=loss_fn(z[250:],yy); loss.backward(); opt.step()
        if epoch==1 or epoch%a.eval_every==0 or epoch==pub["num_steps"]:
            # Author validation calls model.eval() but reuses H_1_words saved
            # by the immediately preceding dropout-active TRAIN pass.
            Hwords_train=model.H_1_words.detach().clone()
            model.eval()
            with torch.no_grad():
                Rd,zd=model.dev_repr_logits(Xdv,Hwords_train); pdv=torch.softmax(zd,1).cpu().numpy()
            m=metrics(ydv,pdv); key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
            hist.append({"epoch":epoch,"loss":float(loss.detach().cpu()),**m})
            if len(hist)%25==0 or epoch==pub["num_steps"]: pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
            if epoch==1 or epoch%max(1,pub["num_steps"]//50)==0:
                print(f"epoch={epoch:05d} loss={loss.item():.4f} dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")
            if key>best:
                best=key; best_epoch=epoch
                torch.save({"model_state_dict":model.state_dict(),"H_1_words":Hwords_train.cpu(),
                            "epoch":epoch,"published_hparams":pub},ckpt)
    pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
    state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"]); model.eval()
    Hwords=state["H_1_words"].to(device)
    with torch.no_grad():
        # Deterministic TRAIN embedding for downstream fusion; DEV inference
        # uses the exact stored training-word state from the best epoch.
        H_eval=model.h1(conv0); Rt=A@H_eval; zt=model.out(Rt)[250:]
        Rd,zd=model.dev_repr_logits(Xdv,Hwords)
        ptr=torch.softmax(zt,1).cpu().numpy(); pdv=torch.softmax(zd,1).cpu().numpy()
    mt,md=metrics(ytr,ptr),metrics(ydv,pdv)
    np.savez_compressed(out/"train_text_embeddings.npz",participant_ids=tr.participant_id.to_numpy(int),labels=ytr,embedding=Rt[250:].cpu().numpy().astype(np.float32),probability=ptr[:,1].astype(np.float32))
    np.savez_compressed(out/"dev_text_embeddings.npz",participant_ids=dv.participant_id.to_numpy(int),labels=ydv,embedding=Rd.cpu().numpy().astype(np.float32),probability=pdv[:,1].astype(np.float32))
    with (out/"vectorizer.pkl").open("wb") as f: pickle.dump(v,f)
    torch.save({"model_state_dict":model.state_dict(),"H1_words":Hwords.cpu(),"best_epoch":best_epoch,"published_hparams":pub},out/"inference_state.pt")
    ours=np.array(sorted(v.vocabulary_,key=v.vocabulary_.get))
    pd.DataFrame({"term":ours}).to_csv(out/"top250_terms.csv",index=False)

    # Diagnostic only: compare independently reconstructed TRAIN vocabulary
    # with the locked published teacher vectorizer. It never changes training.
    teacher_vt=src/"model/Participant/vtzer_inductgcn[250].pkl"
    vocab_audit={"teacher_vectorizer_found":teacher_vt.exists()}
    if teacher_vt.exists():
        with teacher_vt.open("rb") as f: tv=pickle.load(f)
        theirs=set(tv.vocabulary_.keys()); ourset=set(v.vocabulary_.keys()); inter=ourset & theirs
        vocab_audit.update({"our_vocab":len(ourset),"teacher_vocab":len(theirs),"overlap":len(inter),
                            "overlap_fraction":len(inter)/max(1,len(ourset)),
                            "jaccard":len(inter)/max(1,len(ourset|theirs))})
    protocol={"mode":"hard-label text branch pretraining","architecture":"InducT-style original top250; independently trained weights","train_participants":107,"dev_participants":34,
              "participant_440_excluded":True,"teacher_checkpoint_loaded":False,"teacher_vectorizer_loaded":False,
              "published_optuna_hparams_reused":True,"author_stateful_H1_words_behavior":True,
              "vocabulary_overlap_audit":vocab_audit,"best_epoch":best_epoch,"test_opened":False}
    (out/"metrics.json").write_text(json.dumps({"train":mt,"dev34":md,"protocol":protocol},indent=2)+"\n")
    print("\nBEST TEXT-BRANCH DEV-34:",json.dumps(md,indent=2)); print("best_epoch:",best_epoch,"| TEST CLOSED.")
if __name__=="__main__": main()
