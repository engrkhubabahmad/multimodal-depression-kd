"""Fetch and verify only the pinned Idiap architecture source when absent locally."""
import hashlib
import urllib.request
from pathlib import Path

COMMIT = 'de8f477619ce61a2a65640a0735aa8dfe659ac59'
MAIN_BLOB = 'fffbcaf8ddb8f7736f65040c623dbd936771270d'


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
    data = main.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != MAIN_BLOB:
        main.unlink(missing_ok=True)
        raise ValueError('Pinned Idiap source failed Git blob verification')
    return target
