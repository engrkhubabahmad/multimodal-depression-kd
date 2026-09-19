"""USSD ComParE16 audio teacher: frozen author audit -> TRAIN-only adaptation -> KD export."""

AUTHOR_REPO = "https://github.com/vijaysumaravi/USSD-depression.git"
AUTHOR_COMMIT = "c3e68649153004ed2174878a1c54c716ad26cfd7"
EXP_NAME = "cnn_lstm_feature_compare16_delta_feat_dim_384_batch_20_lr_0.003_wdecay_0_lrf_2_alpha_4e-06"
RUN4_REL = f"speaker_disentanglement/best_model/{EXP_NAME}/model/4"
CHECKPOINT_NAME = "md_35_epochs.pth"
AUTHOR_DEV35_MACRO_F1 = 0.8011363636363635
SEGMENT_FRAMES = 384
FREQ_BINS = 130
AUTHOR_TRAIN_CROP_FRAMES = 6662
EXCLUDED_DEV_IDS = {440}
