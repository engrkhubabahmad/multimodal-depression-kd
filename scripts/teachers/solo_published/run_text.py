from pathlib import Path
import argparse,importlib.util,pickle,sys
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm
from .common import DEV_IDS,labels,metrics,save_json,transcript

def load_author_main(path):
    spec=importlib.util.spec_from_file_location('idiap_author_main',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def checkpoint_probabilities(model,row_indices):
    """Run the author graph using TF-IDF rows saved inside the checkpoint."""
    model=model.cpu().eval(); vocab=len(model._vocab); exact=model.A_B[:,:vocab].cpu(); exact=exact[row_indices]
    n=len(row_indices); h0=torch.cat([torch.eye(vocab),exact]); adjacency=torch.zeros((n,vocab+n)); adjacency[:,:vocab]=exact; adjacency[:,vocab:]=torch.eye(n)
    with torch.no_grad():
        h1_docs=model.get_H_1(adjacency@h0); h1=torch.cat([model.H_1_words[:vocab].cpu(),h1_docs]); return torch.softmax(model.node_emb2out(adjacency@h1),dim=-1).numpy(),exact.numpy()

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--daic-root',required=True); p.add_argument('--dev-csv'); p.add_argument('--source-root',default='/content/solo_teacher_sources/bias_in_daic-woz'); p.add_argument('--output',required=True); p.add_argument('--exclude',type=int,nargs='*',default=[440]); a=p.parse_args(argv)
    root=Path(a.daic_root); src=Path(a.source_root); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    split=Path(a.dev_csv) if a.dev_csv else next(root.rglob('dev_split_Depression_AVEC2017.csv')); ymap=labels(split); excluded=set(a.exclude)
    ids=[i for i in DEV_IDS if i not in excluded]
    if set(ids)-set(ymap): raise ValueError(f'Missing labels: {sorted(set(ids)-set(ymap))}')
    docs=[]; rows=[]
    for pid in tqdm(ids,desc='Text raw preprocessing',colour='green'):
        path,df=transcript(root,pid); keep=df[(df.speaker.astype(str).str.casefold()=='participant') & ~df.value.astype(str).str.contains('scrubbed_entry',case=False,na=False)]; doc=' '.join(keep.value.astype(str)).strip();
        if not doc: raise ValueError(f'{pid}: empty participant transcript')
        docs.append(doc); rows.append({'participant_id':pid,'label':ymap[pid],'utterances':len(keep),'transcript_path':str(path)})
    author=load_author_main(src/'main.py'); author.DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with (src/'model/Participant/vtzer_inductgcn[250].pkl').open('rb') as f: vectorizer=pickle.load(f)
    state=torch.load(src/'model/Participant/model_inductgcn[250].pkl',map_location=author.DEVICE,weights_only=False)
    model=author.InducTGCN(state['embedding_dim'],state['classes_'],0,vectorizer); model.load_state_dict(state['model_state_dict']); model.A_B=state['A_dev']; model.classes_=state['classes_']; model.to(author.DEVICE); model.Conv_0_Test=None
    classes=list(model.classes_); pos=classes.index('positive'); y=np.array([r['label'] for r in rows]); selected=[i for i,pid in enumerate(DEV_IDS) if pid not in excluded]
    all_exact,_=checkpoint_probabilities(model,list(range(len(DEV_IDS)))); all_y=np.array([ymap[i] for i in DEV_IDS]); official_result=metrics(all_y,(all_exact[:,pos]>=.5).astype(int),all_exact[:,pos])
    exact,exact_tfidf=checkpoint_probabilities(model,selected); exact_prob=exact[:,pos]; exact_pred=(exact_prob>=.5).astype(int); exact_result=metrics(y,exact_pred,exact_prob)
    model.Conv_0_Test=None; raw=model(docs).detach().cpu().numpy()[:,pos]; raw_pred=(raw>=.5).astype(int); raw_result=metrics(y,raw_pred,raw)
    raw_tfidf=vectorizer.transform(docs).toarray(); denom=np.linalg.norm(raw_tfidf,axis=1)*np.linalg.norm(exact_tfidf,axis=1); cosine=np.divide((raw_tfidf*exact_tfidf).sum(1),denom,out=np.zeros_like(denom),where=denom>0)
    for r,q,z,rq,rz,c in zip(rows,exact_prob,exact_pred,raw,raw_pred,cosine): r.update(prob_depressed=float(q),prediction=int(z),raw_reconstructed_probability=float(rq),raw_reconstructed_prediction=int(rz),raw_to_author_tfidf_cosine=float(c))
    pd.DataFrame(rows).to_csv(out/'dev_predictions.csv',index=False); result={'model':'idiap/participant-induct-gcn-top250','official_checkpoint_dev35':official_result,'local_checkpoint_dev34':exact_result,'raw_reconstructed_dev34':raw_result,'mean_raw_to_author_tfidf_cosine':float(cosine.mean()),'excluded_ids':sorted(excluded),'test_opened':False}; save_json(out/'metrics.json',result)
    print('Exact author checkpoint | DEV-35:',official_result); print('Exact author checkpoint | DEV-34:',exact_result); print('Raw reconstruction      | DEV-34:',raw_result); print('Mean raw/author TF-IDF cosine:',result['mean_raw_to_author_tfidf_cosine']); return result

if __name__=='__main__': main()
