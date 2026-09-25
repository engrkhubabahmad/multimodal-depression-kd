"""Build a small repair set and merge repaired rows without rerunning valid windows."""
import json
from pathlib import Path

from .evaluate_stepaudio2_val import explicit_yes_no


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audio_key(row):
    audios = row.get('audios')
    if not isinstance(audios, list) or len(audios) != 1:
        raise ValueError('Every generation must contain exactly one audio path')
    return str(Path(audios[0]).resolve())


def prepare_repair_dataset(dataset, generations, output):
    """Write only rows whose saved generation lacks one unique explicit label."""
    source = read_jsonl(dataset)
    saved = read_jsonl(generations)
    source_by_key = {audio_key(row): row for row in source}
    saved_by_key = {audio_key(row): row for row in saved}
    if len(source_by_key) != len(source) or len(saved_by_key) != len(saved):
        raise ValueError('Duplicate audio paths in source dataset or saved generations')
    if set(source_by_key) != set(saved_by_key):
        raise ValueError('Saved generations do not match the complete source window set')
    missing = [key for key, row in saved_by_key.items() if explicit_yes_no(row.get('response')) is None]
    repair_rows = [source_by_key[key] for key in missing]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in repair_rows)
    if output.exists() and output.read_text() != content:
        raise ValueError(f'Existing repair dataset differs: {output}')
    output.write_text(content)
    print(f'Saved repair subset: {len(repair_rows)} of {len(source)} windows; valid saved generations reused.')
    return missing


def merge_repaired_predictions(dataset, generations, repairs, output):
    """Merge explicit repaired labels with saved valid generations into a new file."""
    source = read_jsonl(dataset)
    saved = read_jsonl(generations)
    repair_rows = read_jsonl(repairs)
    source_keys = {audio_key(row) for row in source}
    saved_by_key = {audio_key(row): row for row in saved}
    repair_by_key = {audio_key(row): row for row in repair_rows}
    if len(saved_by_key) != len(saved) or len(repair_by_key) != len(repair_rows):
        raise ValueError('Duplicate audio paths in generations or repair results')
    missing = {key for key, row in saved_by_key.items() if explicit_yes_no(row.get('response')) is None}
    if not missing or set(repair_by_key) != missing:
        raise ValueError(f'Repair coverage differs: expected {len(missing)} missing windows, found {len(repair_by_key)}')
    if any(explicit_yes_no(row.get('response')) is None for row in repair_by_key.values()):
        raise ValueError('At least one repaired window still has no unique explicit Yes/No answer')
    if source_keys != set(saved_by_key):
        raise ValueError('Saved generations do not cover the source dataset exactly')
    merged = [repair_by_key.get(audio_key(row), saved_by_key[audio_key(row)]) for row in saved]
    if any(explicit_yes_no(row.get('response')) is None for row in merged):
        raise ValueError('Merged generations still contain an ambiguous answer')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in merged)
    if output.exists() and output.read_text() != content:
        raise ValueError(f'Existing merged file differs: {output}')
    output.write_text(content)
    print(f'Merged {len(repair_by_key)} repaired rows; reused {len(saved) - len(repair_by_key)} saved rows.')
    return output
