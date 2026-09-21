"""Compact branch-pretrained participant student v3.

No-KD branch pretraining uses hard labels only.
Text follows the published InducT-GCN mechanism with independently trained weights.
Audio follows teacher-compatible ComParE16 segmentation/normalization with a smaller recurrent encoder.
"""
TEXT_DIM=64
AUDIO_HIDDEN=128
SEGMENT_FRAMES=384
