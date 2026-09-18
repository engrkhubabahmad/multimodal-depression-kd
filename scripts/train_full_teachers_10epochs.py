"""Run inside an existing Colab runtime: 10-epoch full-model teacher training.
No dependency installation, frozen-feature extraction or OOF loop.
Uses the existing Hugging Face download cache and epoch-boundary checkpoints.
Raw inputs are still necessary for encoder gradients.
"""
from IPython.display import display


from pathlib import Path
if not Path('/content/drive/MyDrive').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
from pathlib import Path
import os, re, json, hashlib, gc, math, random, unicodedata, shutil
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from torch import nn
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import expit
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from huggingface_hub import model_info
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, log_loss

assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU, then rerun.'
device = torch.device('cuda')
DATA_ROOT = Path('/content/drive/MyDrive/DAIC_WOZ'); RUN_ROOT = DATA_ROOT / 'experiments'
assert DATA_ROOT.is_dir(), 'Check DATA_ROOT'
SEED = 103; EPOCHS = 10
RUN_NAME = 'seed_103_full_10epochs_v1'
FULL_ROOT = RUN_ROOT / 'teachers' / 'full_finetune_v1' / RUN_NAME
FULL_ROOT.mkdir(parents=True, exist_ok=True)
ENCODER_LR = 1e-5; HEAD_LR = 1e-4; WEIGHT_DECAY = 0.01
ACCUM_PARTICIPANTS = 4; CHUNKS_PER_PARTICIPANT = 4
FEATURE_CFG = dict(max_segments=128, audio_seconds=4.0, min_audio_seconds=0.5,
                   sample_rate=16000, text_tokens=254)
MODEL_NAMES = {'audio':'facebook/wav2vec2-base-960h',
               'text':'sentence-transformers/all-MiniLM-L6-v2'}
print('GPU:', torch.cuda.get_device_name(0)); print('Run folder:', FULL_ROOT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def read_split(filename, labelled):
    path = DATA_ROOT / 'metadata' / filename
    df = pd.read_csv(path); df.columns = df.columns.str.strip().str.lower()
    assert 'participant_id' in df, f'Missing participant_id in {filename}'
    ids = pd.to_numeric(df['participant_id'], errors='raise')
    assert ids.notna().all() and (ids % 1 == 0).all(), 'Invalid participant IDs'
    assert not ids.duplicated().any() and len(ids), 'Empty or duplicate split IDs'
    result = pd.DataFrame({'participant_id': ids.astype(int)})
    if labelled:
        assert 'phq8_binary' in df, f'Missing labels in {filename}'
        labels = pd.to_numeric(df['phq8_binary'], errors='raise')
        assert labels.notna().all() and labels.isin([0, 1]).all(), 'Invalid binary labels'
        assert labels.nunique() == 2, 'Both classes required'
        result['label'] = labels.astype(int)
    return result, sha256_file(path)

train, train_hash = read_split('train_split_Depression_AVEC2017.csv', True)
dev, dev_hash = read_split('dev_split_Depression_AVEC2017.csv', True)
test_ids, test_hash = read_split('test_split_Depression_AVEC2017.csv', False)
sets = [set(d.participant_id) for d in (train, dev, test_ids)]
assert all(not sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)), 'Participant leakage'
manifest = pd.concat([train.assign(split='train'), dev.assign(split='dev')], ignore_index=True)

# Explicit data-quality exclusion confirmed by the dataset owner.
EXCLUDED_PARTICIPANTS = {440: 'Original participant files corrupted; excluded by user decision'}
official_counts = manifest.groupby('split').size().to_dict()
excluded_rows = manifest.loc[manifest.participant_id.isin(EXCLUDED_PARTICIPANTS), ['participant_id', 'split']].copy()
excluded_rows['reason'] = excluded_rows.participant_id.map(EXCLUDED_PARTICIPANTS)
manifest = manifest.loc[~manifest.participant_id.isin(EXCLUDED_PARTICIPANTS)].reset_index(drop=True)
effective_counts = manifest.groupby('split').size().to_dict()
assert set(effective_counts) == {'train', 'dev'}, 'A labelled split is empty after exclusions'
assert manifest.groupby('split').label.nunique().eq(2).all(), 'Both classes required after exclusions'
exclusion_report = {
    'requested_exclusions': {str(k): v for k, v in EXCLUDED_PARTICIPANTS.items()},
    'excluded': excluded_rows.to_dict(orient='records'),
    'official_counts': {k: int(v) for k, v in official_counts.items()},
    'effective_counts': {k: int(v) for k, v in effective_counts.items()},
}
print('Documented exclusions:', exclusion_report['excluded'])
print('Official counts:', official_counts, '| Used counts:', effective_counts)

audio_index, text_index = {}, {}
for base, dirs, files in os.walk(DATA_ROOT):
    dirs[:] = [d for d in dirs if d not in {'experiments', 'processed', '.git', '__pycache__'}]
    for name in files:
        a = re.fullmatch(r'(\d+)_AUDIO\.wav', name, re.I)
        t = re.fullmatch(r'(\d+)_TRANSCRIPT\.(?:csv|txt)', name, re.I)
        if a: audio_index.setdefault(int(a.group(1)), []).append(Path(base) / name)
        if t: text_index.setdefault(int(t.group(1)), []).append(Path(base) / name)

problems = []
for row in manifest.itertuples():
    pid = int(row.participant_id)
    for kind, index in [('audio', audio_index), ('transcript', text_index)]:
        paths = index.get(pid, [])
        if len(paths) != 1:
            detail = '; '.join(str(p.relative_to(DATA_ROOT)) for p in paths)
            problems.append(f'{pid} ({row.split}): {kind} matches={len(paths)} {detail}')
assert not problems, 'Additional file coverage problems (not automatically excluded): ' + '; '.join(problems[:20])

manifest['audio_path'] = [str(audio_index[p][0]) for p in manifest.participant_id]
manifest['transcript_path'] = [str(text_index[p][0]) for p in manifest.participant_id]
print(manifest.groupby('split').label.agg(['count', 'sum']).rename(columns={'sum':'depressed'}))
print('Split overlap: none. Test participants reserved:', len(test_ids))


def clean_response(value):
    value = unicodedata.normalize('NFC', str(value)).replace('\x00', ' ')
    return re.sub('\\s+', ' ', value).strip()

def read_turns(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = df.columns.str.strip().str.lower()
    if not {'start_time', 'stop_time', 'speaker', 'value'} <= set(df.columns):
        df = pd.read_csv(path, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
    required = {'start_time', 'stop_time', 'speaker', 'value'}
    assert required <= set(df.columns), f'Unexpected transcript columns: {list(df.columns)}'
    df = df.loc[df.speaker.astype(str).str.strip().str.lower().eq('participant')].copy()
    df['value'] = df['value'].fillna('').map(clean_response)
    df = df.loc[df.value.ne('') & ~df.value.str.contains('scrubbed|redacted', case=False, regex=True)]
    for col in ['start_time', 'stop_time']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    assert len(df), f'No usable participant turns in {Path(path).name}'
    assert np.isfinite(df[['start_time', 'stop_time']].to_numpy()).all(), 'Nonfinite timestamps'
    assert ((df.start_time >= 0) & (df.stop_time > df.start_time)).all(), 'Invalid timestamps'
    return df.sort_values('start_time').reset_index(drop=True)

def uniform_subset(items, limit):
    if len(items) <= limit:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, limit, dtype=int)]

def audio_windows(turns, seconds, minimum):
    windows = []
    for row in turns.itertuples():
        start, stop = (float(row.start_time), float(row.stop_time))
        while start < stop:
            end = min(start + seconds, stop)
            if end - start >= minimum:
                windows.append((start, end))
            start = end
    assert windows, 'No sufficiently long participant speech windows'
    return uniform_subset(windows, FEATURE_CFG['max_segments'])

turns_by_id = {int(r.participant_id): read_turns(r.transcript_path) for r in manifest.itertuples()}
revision_file = FULL_ROOT / 'encoder_revisions.json'
if revision_file.exists():
    revisions = json.loads(revision_file.read_text())
else:
    revisions = {name: model_info(name).sha for name in MODEL_NAMES.values()}
    save_json(revision_file, revisions)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES['text'], revision=revisions[MODEL_NAMES['text']])
audio_processor = AutoFeatureExtractor.from_pretrained(MODEL_NAMES['audio'], revision=revisions[MODEL_NAMES['audio']])
inputs_by_id = {}
for row in tqdm(list(manifest.itertuples()), desc='Prepare raw inputs'):
    pid = int(row.participant_id); turns = turns_by_id[pid]
    windows = audio_windows(turns, FEATURE_CFG['audio_seconds'], FEATURE_CFG['min_audio_seconds'])
    with sf.SoundFile(row.audio_path) as wav:
        assert turns.stop_time.max() <= len(wav)/wav.samplerate + 0.1, f'{pid}: transcript exceeds recording duration'
    chunks = []
    for response in turns.value:
        ids = tokenizer.encode(response, add_special_tokens=False)
        for start in range(0, len(ids), FEATURE_CFG['text_tokens']):
            tokens = tokenizer.build_inputs_with_special_tokens(ids[start:start+FEATURE_CFG['text_tokens']])
            chunks.append(tokens)
    chunks = uniform_subset(chunks, FEATURE_CFG['max_segments'])
    assert chunks and windows, f'{pid}: no usable chunks'
    inputs_by_id[pid] = {'audio_path': row.audio_path, 'windows': windows, 'tokens': chunks}

config = dict(protocol='full_finetune_v1', epochs=EPOCHS, seed=SEED, feature_config=FEATURE_CFG,
              chunks_per_participant=CHUNKS_PER_PARTICIPANT, accumulation=ACCUM_PARTICIPANTS,
              encoder_lr=ENCODER_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY,
              model_names=MODEL_NAMES, revisions=revisions, exclusions=exclusion_report,
              threshold=0.5, selection='minimum development log_loss', all_encoder_parameters_trainable=True,
              test_used=False, oof=False, audio_specaugment=False, audio_layerdrop=0.0,
              training_sampling='random chunks per participant per epoch', evaluation_sampling='fixed uniform chunks',
              splits={'train':train_hash, 'dev':dev_hash, 'test_ids':test_hash},
              versions={p:version(p) for p in ['torch','transformers','numpy','pandas','scikit-learn','soundfile','scipy']})
source_hashes = {str(r.participant_id):{'audio':sha256_file(r.audio_path), 'transcript':sha256_file(r.transcript_path)}
                 for r in tqdm(list(manifest.itertuples()), desc='Fingerprint sources')}
config['source_hashes'] = source_hashes
config_path = FULL_ROOT / 'config.json'
if config_path.exists():
    assert json.loads(config_path.read_text()) == config, 'Configuration/data changed: choose a NEW RUN_NAME.'
else:
    save_json(config_path, config)
manifest.to_csv(FULL_ROOT / 'train_dev_manifest.csv', index=False)
save_json(FULL_ROOT / 'exclusions.json', exclusion_report)
train_rows = list(manifest.loc[manifest.split.eq('train')].itertuples())
dev_rows = list(manifest.loc[manifest.split.eq('dev')].itertuples())


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def select_chunks(items, pid, epoch, training):
    k = min(CHUNKS_PER_PARTICIPANT, len(items))
    if training:
        rng = np.random.default_rng(SEED + 100000*epoch + int(pid))
        return [items[i] for i in sorted(rng.choice(len(items), size=k, replace=False))]
    return uniform_subset(items, k)

def participant_inputs(pid, modality, epoch=0, training=False):
    entry = inputs_by_id[int(pid)]; result = []
    if modality == 'text':
        for ids in select_chunks(entry['tokens'], pid, epoch, training):
            result.append({'input_ids':torch.tensor([ids], dtype=torch.long, device=device),
                           'attention_mask':torch.ones((1,len(ids)), dtype=torch.long, device=device)})
    else:
        with sf.SoundFile(entry['audio_path']) as wav:
            rate = wav.samplerate
            for start, end in select_chunks(entry['windows'], pid, epoch, training):
                wav.seek(min(round(start*rate), len(wav)))
                x = wav.read(max(1, round((end-start)*rate)), dtype='float32', always_2d=True).mean(axis=1)
                assert len(x) and np.isfinite(x).all(), f'{pid}: invalid audio'
                if rate != FEATURE_CFG['sample_rate']:
                    divisor = math.gcd(rate, FEATURE_CFG['sample_rate'])
                    x = resample_poly(x, FEATURE_CFG['sample_rate']//divisor, rate//divisor).astype(np.float32)
                if len(x) < FEATURE_CFG['min_audio_seconds']*FEATURE_CFG['sample_rate']: continue
                if np.max(np.abs(x)) < 1e-7: continue
                values = audio_processor(x, sampling_rate=FEATURE_CFG['sample_rate'], return_tensors='pt').input_values
                result.append({'input_values':values.to(device)})
    assert result, f'{pid}: selected chunks have no usable input; inspect this participant'
    return result

class FullTeacher(nn.Module):
    def __init__(self, modality):
        super().__init__(); self.modality = modality
        name = MODEL_NAMES[modality]
        self.encoder = AutoModel.from_pretrained(name, revision=revisions[name], use_safetensors=True)
        if modality == 'audio':
            self.encoder.config.apply_spec_augment = False
            self.encoder.config.mask_time_prob = 0.0
            self.encoder.config.mask_feature_prob = 0.0
            self.encoder.config.layerdrop = 0.0
        self.encoder.requires_grad_(True)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        dim = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,1))

    def forward(self, chunks):
        vectors = []
        for batch in chunks:
            hidden = self.encoder(**batch).last_hidden_state
            if self.modality == 'text':
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            else:
                pooled = hidden.mean(1)  # Unpadded single audio window.
            vectors.append(pooled)
        participant = torch.stack(vectors).mean(0)
        return self.head(participant).reshape(())

def evaluate(model, rows, modality):
    model.eval(); records = []
    with torch.inference_mode():
        for row in rows:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                z = model(participant_inputs(row.participant_id, modality)).float().item()
            records.append({'participant_id':int(row.participant_id), 'label':int(row.label), 'logit':float(z)})
    df = pd.DataFrame(records)
    assert np.isfinite(df.logit).all(), 'Nonfinite evaluation logits'
    df['probability'] = expit(df.logit); df['prediction'] = df.probability.ge(0.5).astype(int)
    y, p, pred = df.label, df.probability, df.prediction
    scores = {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
              'depressed_f1':float(f1_score(y,pred,zero_division=0)),
              'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
              'auroc':float(roc_auc_score(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1]))}
    return df, scores

def atomic_torch_save(value, path):
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary); temporary.replace(path)

def train_modality(modality):
    seed_all(SEED)
    out = FULL_ROOT / modality; out.mkdir(parents=True, exist_ok=True)
    model = FullTeacher(modality).to(device)
    assert all(p.requires_grad for p in model.parameters()), 'Some model parameters were frozen'
    print(modality, 'trainable parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':ENCODER_LR},
                                  {'params':model.head.parameters(),'lr':HEAD_LR}], weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    counts = np.bincount([int(r.label) for r in train_rows], minlength=2)
    assert (counts>0).all()
    weights = {i:len(train_rows)/(2*counts[i]) for i in [0,1]}
    history = []; best_loss = float('inf'); best_epoch = None; start_epoch = 1
    latest = out / 'last.pt'; best_path = out / 'best.pt'
    if latest.exists():
        # Only load our own trusted checkpoint; includes optimizer and history.
        state = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']); history = state['history']
        best_loss = state['best_loss']; best_epoch = state['best_epoch']; start_epoch = state['epoch']+1
        del state; gc.collect()
        print('Resuming from epoch', start_epoch, flush=True)
    for epoch in range(start_epoch, EPOCHS+1):
        seed_all(SEED+epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        order = np.random.default_rng(SEED+epoch).permutation(len(train_rows))
        loss_total = 0.0; gradient_checked = False; updates = 0
        for position, index in enumerate(tqdm(order, desc=f'{modality} epoch {epoch}/{EPOCHS}')):
            row = train_rows[int(index)]
            group_start = (position//ACCUM_PARTICIPANTS)*ACCUM_PARTICIPANTS
            group_size = min(ACCUM_PARTICIPANTS, len(order)-group_start)
            chunks = participant_inputs(row.participant_id, modality, epoch, True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                z = model(chunks)
                target = torch.tensor(float(row.label), device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z.float(), target)*weights[int(row.label)]
            assert torch.isfinite(loss).item(), f'Nonfinite loss for participant {row.participant_id}'
            scaler.scale(loss/group_size).backward()
            loss_total += float(loss.detach().cpu())
            if not gradient_checked:
                encoder_grad = any(p.grad is not None and torch.isfinite(p.grad).all().item() and torch.count_nonzero(p.grad).item()>0 for p in model.encoder.parameters())
                head_grad = any(p.grad is not None and torch.isfinite(p.grad).all().item() and torch.count_nonzero(p.grad).item()>0 for p in model.head.parameters())
                assert encoder_grad and head_grad, 'Gradient check failed: encoder or classifier has no finite nonzero gradient'
                gradient_checked = True
            if (position+1)%ACCUM_PARTICIPANTS == 0 or position+1 == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale: updates += 1
                optimizer.zero_grad(set_to_none=True)
            del chunks, z, loss
        assert updates > 0, 'All optimizer steps were skipped; inspect numerical stability'
        dev_predictions, scores = evaluate(model, dev_rows, modality)
        record = {'epoch':epoch, 'train_loss':loss_total/len(order), 'optimizer_updates':updates, **scores}
        history.append(record); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'{modality} epoch {epoch}:', record, flush=True)
        if scores['log_loss'] < best_loss:
            best_loss = scores['log_loss']; best_epoch = epoch
            atomic_torch_save({'model':model.state_dict(),'epoch':epoch,'config':config},best_path)
        atomic_torch_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                          'epoch':epoch,'history':history,'best_loss':best_loss,'best_epoch':best_epoch},latest)
    # Both last-epoch and best-development checkpoints are retained.
    final_predictions, final_scores = evaluate(model,dev_rows,modality)
    final_path = out/'last_dev_predictions.csv'; final_predictions.to_csv(final_path,index=False)
    print_save_classification_report(final_path)
    state = torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); del state
    best_predictions, best_scores = evaluate(model,dev_rows,modality)
    best_file = out/'best_dev_predictions.csv'; best_predictions.to_csv(best_file,index=False)
    print_save_classification_report(best_file)
    training_predictions, training_scores = evaluate(model,train_rows,modality)
    train_file = out/'best_train_in_sample_predictions.csv'; training_predictions.to_csv(train_file,index=False)
    print_save_classification_report(train_file)
    save_json(out/'metrics.json',{'best_epoch':best_epoch,'last_dev':final_scores,'best_dev':best_scores,
                                  'train_in_sample':training_scores,'oof_available':False})
    result = {'modality':modality,'best_epoch':best_epoch,**best_scores}
    del model, optimizer, scaler; gc.collect(); torch.cuda.empty_cache()
    return result


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def print_save_classification_report(predictions_path):
    path = Path(predictions_path)
    df = pd.read_csv(path)
    assert {'participant_id', 'label', 'probability'} <= set(df.columns), f'Missing columns: {path}'
    assert len(df) and not df.participant_id.duplicated().any(), 'Expected one row per participant'
    assert df.label.isin([0, 1]).all(), 'Invalid labels'
    assert df.probability.notna().all() and df.probability.between(0, 1).all(), 'Invalid probabilities'
    y = df.label.astype(int).to_numpy()
    pred = df.probability.ge(0.5).astype(int).to_numpy()
    names = ['Non-depressed (0)', 'Depressed (1)']
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1]),
                      index=['Actual ' + n for n in names],
                      columns=['Predicted ' + n for n in names])
    kwargs = dict(labels=[0, 1], target_names=names, zero_division=0)
    report = classification_report(y, pred, digits=4, **kwargs)
    cr = pd.DataFrame(classification_report(y, pred, output_dict=True, **kwargs)).T
    print(f'\n{path.parent.parent.name} / {path.parent.name} / {path.stem} | threshold=0.5 | n={len(df)}')
    print('CM: rows = actual, columns = predicted')
    print(cm.to_string())
    print('\nCR:')
    print(report)
    stem = path.stem.removesuffix('_predictions')
    cm.to_csv(path.with_name(stem + '_confusion_matrix.csv'))
    cr.to_csv(path.with_name(stem + '_classification_report.csv'))
    path.with_name(stem + '_classification_report.txt').write_text(report)
    return cm, cr


results=[]
for modality in ['audio','text']:
    results.append(train_modality(modality))
    pd.DataFrame(results).to_csv(FULL_ROOT/'teacher_summary.csv',index=False)
display(pd.DataFrame(results))
print('Saved full-model teachers:',FULL_ROOT)
