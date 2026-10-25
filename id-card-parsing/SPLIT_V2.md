# Expanded ID card dataset split

The expanded manifest contains 200 annotated images: 75 train, 25 validation, and 100 test. The original 60/20/20 assignments and rotation settings are unchanged and archived in `splits_v1.csv`.

The 100 newly annotated images are assigned deterministically using Python Random seed 42 after sorting image IDs: 15 train, 5 validation, and 80 test. Augmentations belong only in training. Prepared outputs use `data/id_cards/v2` to preserve the original dataset.

Each new image currently has its own group ID and rotation_quarters=0. Confirm specimen independence and review upright rectified cards before training or evaluation; these settings are provisional. If multiple scans show the same specimen, put them in the same split and rebalance by group.

Evaluate the existing checkpoint before retraining and retain its results. Compare the updated model on the same fixed test set; also report the 80 new test images separately from the 20 previously evaluated images. Do not use test outcomes to tune training settings.
