"""Participant-level ReLiMP-Net student for strict DAIC-WOZ TRAIN/DEV training."""

SEGMENT_CONFIG = {
    "max_segments_per_participant": 128,
    "max_audio_seconds": 10.0,
    "min_audio_seconds": 0.5,
    "sample_rate": 16000,
    "max_text_tokens": 64,
    "max_vocab": 10000,
    "min_token_freq": 2,
    "n_mels": 64,
}
