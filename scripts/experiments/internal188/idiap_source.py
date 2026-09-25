"""Fetch and verify only the pinned Idiap architecture source when absent locally."""
import hashlib
import urllib.request
from pathlib import Path

COMMIT = 'de8f477619ce61a2a65640a0735aa8dfe659ac59'
MAIN_BLOB = 'fffbcaf8ddb8f7736f65040c623dbd936771270d'
CHECKPOINT_BLOB = '4c7cebd72362702efd5cbf3e5521883f17829f60'
VECTORIZER_BLOB = 'e76e5d66226b13b51f2aff4bc58c2484b8e4edd3'
TRAIN_IDS_BLOB = 'fdf3651a8f9a52eeafdb559798cf59e37aef1ab4'
DEV_IDS_BLOB = '4ce5e8b091f45d6fc93e831965c26dd7d8520c76'


def checked_file(path, expected_blob):
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != expected_blob: raise ValueError(f'Pinned Idiap artifact failed Git blob verification: {path}')
    return path


def ensure_source(preferred):
    preferred = Path(preferred)
    if (preferred / 'main.py').is_file():
        return preferred
    target = Path('/content/idiap_bias_in_daic_woz')
    target.mkdir(parents=True, exist_ok=True)
    main = target / 'main.py'
    if not main.is_file():
        url = f'https://raw.githubusercontent.com/idiap/bias_in_daic-woz/{COMMIT}/main.py'
        print('Fetching pinned Idiap architecture source:', COMMIT[:12], flush=True)
        urllib.request.urlretrieve(url, main)
    try: checked_file(main, MAIN_BLOB)
    except ValueError:
        main.unlink(missing_ok=True)
        raise
    return target


def ensure_published_weights():
    target = Path('/content/idiap_bias_in_daic_woz/model/Participant')
    target.mkdir(parents=True, exist_ok=True)
    artifacts = [('model_inductgcn[250].pkl', CHECKPOINT_BLOB),
                 ('vtzer_inductgcn[250].pkl', VECTORIZER_BLOB)]
    for name, blob in artifacts:
        path = target / name
        if not path.is_file():
            url = f'https://raw.githubusercontent.com/idiap/bias_in_daic-woz/{COMMIT}/model/Participant/{name.replace("[", "%5B").replace("]", "%5D")}'
            urllib.request.urlretrieve(url, path)
        try: checked_file(path, blob)
        except ValueError:
            path.unlink(missing_ok=True)
            raise
    return target / artifacts[0][0], target / artifacts[1][0]


def ensure_frozen_export_source():
    """Stage the pinned checkpoint and original participant order in local Colab storage."""
    root = Path('/content/idiap_bias_in_daic_woz')
    ensure_source(root / '_download_if_missing')
    ensure_published_weights()
    for name, blob in [('train_IDS.txt', TRAIN_IDS_BLOB),
                       ('dev_IDS.txt', DEV_IDS_BLOB)]:
        path = root / 'data' / 'AVEC_16_data' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            url = f'https://raw.githubusercontent.com/idiap/bias_in_daic-woz/{COMMIT}/data/AVEC_16_data/{name}'
            urllib.request.urlretrieve(url, path)
        try: checked_file(path, blob)
        except ValueError:
            path.unlink(missing_ok=True)
            raise
    checked_file(root / 'main.py', MAIN_BLOB)
    return root
