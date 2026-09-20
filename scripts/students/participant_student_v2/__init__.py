"""Rich, compact participant-level student.

Audio: cached teacher-compatible ComParE16 sequences.
Text: independently fitted TRAIN-only top-250 TF-IDF.
No teacher predictions are used by the no-KD baseline.
"""

SEGMENT_FRAMES=384
MAX_AUDIO_SEGMENTS=32
TEXT_FEATURES=250
