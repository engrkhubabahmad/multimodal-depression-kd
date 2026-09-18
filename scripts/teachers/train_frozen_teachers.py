"""Frozen audio/text teacher pipeline for DAIC-WOZ.
Generated from notebooks/02_train_audio_text_teachers.ipynb.
Run in Colab after cloning the repository.
"""
from IPython.display import display


# ---- teacher-004 ----
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, re, json, hashlib, gc, math, time, platform, unicodedata
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd

DATA_ROOT = Path('/content/drive/MyDrive/DAIC_WOZ')
RUN_ROOT = DATA_ROOT / 'experiments'
assert DATA_ROOT.is_dir(), f'Dataset folder not found: {DATA_ROOT}'
RUN_ROOT.mkdir(parents=True, exist_ok=True)
SEEDS = [103]  # Later use [103, 104, 105, 106, 107]; report all runs.
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 16
# Known-good frozen baseline augmentation.
AUG_COPIES = 2
AUG_KEEP_FRACTION = 0.8
FEATURE_CFG = dict(pipeline_version=2, text_cleanup="nfc_whitespace_v1", max_segments=128, audio_seconds=10.0,
                   min_audio_seconds=0.5, sample_rate=16000, text_tokens=254,
                   audio_model='facebook/wav2vec2-base-960h',
                   text_model='sentence-transformers/all-MiniLM-L6-v2')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'  # Transformers uses PyTorch; classifier heads use TensorFlow later.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('Output folder:', RUN_ROOT)


# ---- teacher-006 ----
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


# ---- teacher-008 ----
def clean_response(value):
    # Preserve words, negation, punctuation, case, repetitions and annotations.
    value = unicodedata.normalize('NFC', str(value)).replace('\x00', ' ')
    return re.sub(r'\s+', ' ', value).strip()

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
    if len(items) <= limit: return items
    return [items[i] for i in np.linspace(0, len(items)-1, limit, dtype=int)]

def audio_windows(turns, seconds, minimum):
    windows = []
    for row in turns.itertuples():
        start, stop = float(row.start_time), float(row.stop_time)
        while start < stop:
            end = min(start + seconds, stop)
            if end - start >= minimum: windows.append((start, end))
            start = end
    assert windows, 'No sufficiently long participant speech windows'
    return uniform_subset(windows, FEATURE_CFG['max_segments'])

def pool_segments(features):
    x = np.asarray(features, dtype=np.float32)
    assert x.ndim == 2 and len(x) and np.isfinite(x).all(), 'Invalid encoder features'
    return np.concatenate([x.mean(axis=0), x.std(axis=0)]).astype(np.float32)

from huggingface_hub import model_info
# Resolve immutable model revisions once; retries reuse the recorded revisions.
revision_path = RUN_ROOT / 'features' / 'teacher_v1_encoder_revisions.json'
if revision_path.exists():
    revisions = json.loads(revision_path.read_text())
else:
    revisions = {FEATURE_CFG[k]: model_info(FEATURE_CFG[k]).sha for k in ['audio_model', 'text_model']}
    save_json(revision_path, revisions)
assert all(FEATURE_CFG[k] in revisions for k in ['audio_model', 'text_model']), 'Encoder config changed: use a new revision file'
cache_config = dict(FEATURE_CFG, revisions=revisions,
                    packages={p: version(p) for p in ['transformers', 'torch', 'numpy', 'scipy', 'soundfile']})
cache_key = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:16]
CACHE_ROOT = RUN_ROOT / 'features' / ('teacher_v2_' + cache_key)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
save_json(CACHE_ROOT / 'config.json', cache_config)
turns_by_id = {int(r.participant_id): read_turns(r.transcript_path) for r in manifest.itertuples()}
print('Usable participant transcripts:', len(turns_by_id))
print('Cache:', CACHE_ROOT)

def source_signature(row):
    # Full content hashes prevent reuse after source files are replaced.
    return {'audio_sha256': sha256_file(row.audio_path),
            'transcript_sha256': sha256_file(row.transcript_path)}

from tqdm.auto import tqdm
signatures = {int(r.participant_id): source_signature(r)
              for r in tqdm(list(manifest.itertuples()), desc='Fingerprint sources')}

def cache_file(pid, modality): return CACHE_ROOT / f'{int(pid)}_{modality}.npz'

def cached(pid, modality):
    path = cache_file(pid, modality)
    if not path.exists(): return False
    try:
        with np.load(path, allow_pickle=False) as data:
            valid = json.loads(str(data['signature'].item())) == signatures[int(pid)]
            return valid and np.isfinite(data['pooled']).all() and np.isfinite(data['segments']).all()
    except (ValueError, OSError, KeyError): return False

def save_features(pid, modality, features):
    path = cache_file(pid, modality); tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, segments=np.asarray(features, dtype=np.float32),
                            pooled=pool_segments(features),
                            signature=json.dumps(signatures[int(pid)], sort_keys=True))
    tmp.replace(path)


# ---- teacher-quality-code ----
import soundfile as sf

def audit_audio_row(row):
    pid = int(row.participant_id)
    audit_path = CACHE_ROOT / f'{pid}_quality.json'
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        if previous.get('signature') == signatures[pid]: return previous
    turns = turns_by_id[pid]
    windows = audio_windows(turns, FEATURE_CFG['audio_seconds'], FEATURE_CFG['min_audio_seconds'])
    total = silent = near_full_scale = 0; peak = 0.0
    with sf.SoundFile(row.audio_path) as wav:
        rate = wav.samplerate; duration = len(wav)/rate; channels = wav.channels
        assert turns.stop_time.max() <= duration+0.1, f'{pid}: transcript exceeds audio duration'
        for start, end in windows:
            wav.seek(min(round(start*rate), len(wav)))
            samples = wav.read(max(1, round((end-start)*rate)), dtype='float32', always_2d=True)
            assert samples.size and np.isfinite(samples).all(), f'{pid}: empty/nonfinite audio'
            values = np.abs(samples)
            total += values.size; silent += int((values < 1e-7).sum())
            near_full_scale += int((values >= 0.999).sum()); peak = max(peak, float(values.max()))
    ends = np.maximum.accumulate(turns.stop_time.to_numpy())
    overlaps = int((turns.start_time.to_numpy()[1:] < ends[:-1]).sum())
    result = {'participant_id':pid, 'split':str(row.split), 'signature':signatures[pid],
              'sample_rate':int(rate), 'channels':int(channels), 'duration_seconds':float(duration),
              'retained_turns':len(turns), 'sampled_windows':len(windows),
              'near_zero_fraction':float(silent/total), 'near_full_scale_fraction':float(near_full_scale/total),
              'peak_amplitude':peak, 'overlapping_turns':overlaps}
    save_json(audit_path, result)
    return result

quality_report = [audit_audio_row(r) for r in tqdm(list(manifest.itertuples()), desc='Audio quality audit')]
quality_df = pd.DataFrame([{k:v for k,v in row.items() if k != 'signature'} for row in quality_report])
quality_df.to_csv(CACHE_ROOT / 'data_quality.csv', index=False)
flagged = quality_df.loc[(quality_df.near_full_scale_fraction > 0.01) |
                         (quality_df.near_zero_fraction > 0.5) |
                         (quality_df.overlapping_turns > 0) | (quality_df.channels != 1)]
print('Audited participants:', len(quality_df), '| Flagged for review:', len(flagged))
if len(flagged):
    display(flagged)
    print('Quality flags recorded for review; no participant was automatically removed or audio altered.')
print('Audio quality audit completed. This does not verify semantic transcript alignment.')


# ---- teacher-010 ----
import torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Feature extraction device:', device)
if device == 'cpu': print('CPU extraction can be slow. Select a GPU runtime for this stage.')
pending = [r for r in manifest.itertuples() if not cached(r.participant_id, 'audio')]
if pending:
    name = FEATURE_CFG['audio_model']; revision = revisions[name]
    audio_processor = AutoFeatureExtractor.from_pretrained(name, revision=revision)
    audio_encoder = AutoModel.from_pretrained(name, revision=revision, use_safetensors=True).to(device).eval()
    audio_encoder.requires_grad_(False)
    for row in tqdm(pending, desc='Audio participants'):
        pid = int(row.participant_id); features = []
        windows = audio_windows(turns_by_id[pid], FEATURE_CFG['audio_seconds'], FEATURE_CFG['min_audio_seconds'])
        with sf.SoundFile(row.audio_path) as wav:
            rate = wav.samplerate; duration = len(wav) / rate
            assert turns_by_id[pid].stop_time.max() <= duration + 0.1, f'{pid}: transcript exceeds audio duration'
            for start, end in windows:
                wav.seek(min(round(start * rate), len(wav)))
                x = wav.read(max(1, round((end-start) * rate)), dtype='float32', always_2d=True).mean(axis=1)
                assert len(x) and np.isfinite(x).all(), f'{pid}: empty/nonfinite audio'
                if rate != FEATURE_CFG['sample_rate']:
                    divisor = math.gcd(rate, FEATURE_CFG['sample_rate'])
                    x = resample_poly(x, FEATURE_CFG['sample_rate']//divisor, rate//divisor).astype(np.float32)
                if len(x) < FEATURE_CFG['min_audio_seconds'] * FEATURE_CFG['sample_rate']: continue
                if np.max(np.abs(x)) < 1e-7: continue
                inputs = audio_processor(x, sampling_rate=FEATURE_CFG['sample_rate'], return_tensors='pt')
                with torch.inference_mode():
                    h = audio_encoder(input_values=inputs.input_values.to(device)).last_hidden_state
                    features.append(h.mean(dim=1).squeeze(0).cpu().numpy())
        assert features, f'{pid}: no nonsilent usable audio windows'
        save_features(pid, 'audio', features)
    del audio_encoder, audio_processor, inputs, h
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print('Audio caches ready:', sum(cached(p, 'audio') for p in manifest.participant_id))


# ---- teacher-012 ----
pending = [r for r in manifest.itertuples() if not cached(r.participant_id, 'text')]
if pending:
    name = FEATURE_CFG['text_model']; revision = revisions[name]
    text_tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    text_encoder = AutoModel.from_pretrained(name, revision=revision, use_safetensors=True).to(device).eval()
    text_encoder.requires_grad_(False)
    for row in tqdm(pending, desc='Text participants'):
        pid = int(row.participant_id); chunks = []
        for response in turns_by_id[pid]['value']:
            ids = text_tokenizer.encode(response, add_special_tokens=False)
            for start in range(0, len(ids), FEATURE_CFG['text_tokens']):
                tokens = text_tokenizer.build_inputs_with_special_tokens(ids[start:start+FEATURE_CFG['text_tokens']])
                chunks.append({'input_ids': tokens, 'attention_mask': [1]*len(tokens)})
        chunks = uniform_subset(chunks, FEATURE_CFG['max_segments'])
        assert chunks, f'{pid}: no text tokens'
        features = []
        for start in range(0, len(chunks), 16):
            batch = text_tokenizer.pad(chunks[start:start+16], padding=True, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode():
                h = text_encoder(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                features.extend(emb.cpu().numpy())
        save_features(pid, 'text', features)
    del text_encoder, text_tokenizer, batch, h, emb
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print('Text caches ready:', sum(cached(p, 'text') for p in manifest.participant_id))


# ---- teacher-report-code ----
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


# ---- teacher-014 ----
import tensorflow as tf
# Small heads run on CPU, leaving the GPU exclusively to encoder extraction.
tf.config.set_visible_devices([], 'GPU')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score, log_loss, confusion_matrix, brier_score_loss)
from scipy.special import expit

def build_head(input_dim, seed):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return model

def class_weights(y):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    assert (counts > 0).all(), 'Both classes required in each training fold'
    return {i: float(len(y)/(2*counts[i])) for i in (0, 1)}

def augment_training_features(x, y, segments, seed):
    """Known-good feature-view augmentation from the frozen baseline."""
    assert isinstance(AUG_COPIES, int) and AUG_COPIES >= 0
    assert 0 < AUG_KEEP_FRACTION <= 1
    if AUG_COPIES == 0: return x.copy(), y.copy()
    assert segments is not None and len(segments) == len(x) == len(y)
    rng = np.random.default_rng(seed)
    variants = [x.copy()]
    for _ in range(AUG_COPIES):
        rows = []
        for original, chunks in zip(x, segments):
            chunks = np.asarray(chunks, dtype=np.float32)
            assert chunks.ndim == 2 and len(chunks) and np.isfinite(chunks).all()
            assert np.allclose(pool_segments(chunks), original, rtol=1e-4, atol=1e-5), 'Segment/participant order mismatch'
            keep = min(len(chunks), max(2, int(np.ceil(len(chunks)*AUG_KEEP_FRACTION))))
            chosen = np.sort(rng.choice(len(chunks), size=keep, replace=False))
            rows.append(pool_segments(chunks[chosen]))
        variants.append(np.stack(rows))
    xa = np.concatenate(variants).astype(np.float32)
    ya = np.tile(y, AUG_COPIES+1)
    print('Frozen baseline training views:', len(y), '->', len(ya),
          '| original class counts:', np.bincount(np.asarray(y,dtype=int), minlength=2).tolist())
    return xa, ya

def fit_head(x, y, xv, yv, seed, fixed_epochs=None, fit_segments=None):
    scaler = StandardScaler().fit(x)  # Fit on ORIGINAL training participants only.
    xa, ya = augment_training_features(x, y, fit_segments, seed)
    xs = scaler.transform(xa).astype(np.float32)
    model = build_head(x.shape[1], seed)
    callbacks = []; validation = None
    if fixed_epochs is None:
        validation = (scaler.transform(xv).astype(np.float32), yv)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                       restore_best_weights=True)]
    history = model.fit(xs, ya, validation_data=validation, epochs=fixed_epochs or MAX_EPOCHS,
                        batch_size=BATCH_SIZE, class_weight=class_weights(y),
                        callbacks=callbacks, shuffle=True, verbose=0)
    best_epoch = int(np.argmin(history.history['val_loss'])+1) if fixed_epochs is None else fixed_epochs
    return model, scaler, history.history, best_epoch

def logits(model, scaler, x):
    return model(scaler.transform(x).astype(np.float32), training=False).numpy().reshape(-1)

def metrics(y, z):
    p = expit(z); pred = (p >= 0.5).astype(int)
    return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'depressed_f1': float(f1_score(y, pred, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'auroc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}

def save_predictions(path, ids, y, z, fold=None):
    result = pd.DataFrame({'participant_id': ids, 'label': y, 'logit': z,
                           'probability': expit(z), 'prediction': (z >= 0).astype(int)})
    if fold is not None: result['fold'] = fold
    result.to_csv(path, index=False)
    print_save_classification_report(path)

def save_scaler(path, scaler):
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)

def load_segment_lists(modality):
    result = []
    for pid in manifest.participant_id:
        assert cached(pid, modality), f'Missing/stale cache for {pid}'
        with np.load(cache_file(pid, modality), allow_pickle=False) as data:
            result.append(data['segments'].copy())
    return result

def load_matrix(modality):
    rows = []
    for pid in manifest.participant_id:
        assert cached(pid, modality), f'Missing/stale {modality} cache for {pid}'
        with np.load(cache_file(pid, modality), allow_pickle=False) as data:
            rows.append(data['pooled'].copy())
    return np.stack(rows).astype(np.float32)

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
TEACHER_ROOT = RUN_ROOT / 'teachers' / 'frozen_encoder_v2' / run_id
TEACHER_ROOT.mkdir(parents=True, exist_ok=False)
manifest.to_csv(TEACHER_ROOT / 'train_dev_manifest.csv', index=False)
save_json(TEACHER_ROOT / 'exclusions.json', exclusion_report)
save_json(TEACHER_ROOT / 'data_quality.json', quality_report)
config = dict(exclusions=exclusion_report, feature_config=cache_config, cache_path=str(CACHE_ROOT), seeds=SEEDS,
              max_epochs=MAX_EPOCHS, patience=PATIENCE,
              batch_size=BATCH_SIZE, learning_rate=3e-4, threshold=0.5,
              augmentation={'type':'cached_segment_subset', 'copies':AUG_COPIES,
                            'keep_fraction':AUG_KEEP_FRACTION, 'fit_partition_only':True,
                            'raw_waveform_augmentation':False, 'token_augmentation':False},
              split_sha256={'train':train_hash, 'dev':dev_hash, 'test_ids':test_hash},
              packages={p: version(p) for p in ['tensorflow','torch','transformers','numpy','pandas','scikit-learn']},
              python=platform.python_version(), test_used_for_training_or_selection=False)
save_json(TEACHER_ROOT / 'run_config.json', config)
is_train = manifest.split.eq('train').to_numpy()
y_train = manifest.loc[is_train, 'label'].to_numpy()
y_dev = manifest.loc[~is_train, 'label'].to_numpy()
ids_train = manifest.loc[is_train, 'participant_id'].to_numpy()
ids_dev = manifest.loc[~is_train, 'participant_id'].to_numpy()
summary = []

for modality in ['audio', 'text']:
    x = load_matrix(modality); x_train, x_dev = x[is_train], x[~is_train]
    segment_lists = load_segment_lists(modality)
    train_segments = [chunks for chunks, keep in zip(segment_lists, is_train) if keep]
    for seed in SEEDS:
        out = TEACHER_ROOT / modality / f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=False)
        print(f'Training {modality} teacher; seed={seed}; input_dim={x.shape[1]}', flush=True)
        model, scaler, history, epoch = fit_head(x_train, y_train, x_dev, y_dev, seed, fit_segments=train_segments)
        model.save(out / 'teacher.keras'); save_scaler(out / 'scaler.npz', scaler)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        z_dev = logits(model, scaler, x_dev)
        z_train = logits(model, scaler, x_train)
        save_predictions(out / 'dev_predictions.csv', ids_dev, y_dev, z_dev)
        # In-sample predictions are diagnostics only.
        save_predictions(out / 'train_in_sample_predictions.csv', ids_train, y_train, z_train)
        scores = metrics(y_dev, z_dev)
        save_json(out / 'dev_metrics.json', dict(scores, best_epoch=epoch))
        summary.append(dict(modality=modality, seed=seed, evaluation='dev_selection',
                            **{k:v for k,v in scores.items() if k != 'confusion_matrix'}))
        print('Development:', {k:round(v,4) for k,v in scores.items() if isinstance(v,float)}, flush=True)
        del model; gc.collect()

        pd.DataFrame(summary).to_csv(TEACHER_ROOT / 'teacher_summary.csv', index=False)

print('Saved teachers to:', TEACHER_ROOT)
display(pd.DataFrame(summary))
