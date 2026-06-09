# DF-03-01 Dataset Download Check

Note: The following paths are local paths on the author's cloud machine. They are recorded for experiment reproducibility and do not contain access credentials.

## Dataset

Dataset name:

`openvla/modified_libero_rlds`

Cloud dataset path:

`/root/autodl-tmp/datasets/openvla_modified_libero_rlds`

## Download Status

The dataset download finished successfully on the cloud server.

The Hugging Face download process reached:

`Fetching 106 files: 100%`

## Verification Result

Dataset path:

`/root/autodl-tmp/datasets/openvla_modified_libero_rlds`

Verification summary:

| Item                     | Result  |
| ------------------------ | ------- |
| Dataset directory exists | True    |
| Is directory             | True    |
| Total files              | 213     |
| TFRecord files           | 192     |
| Incomplete files         | 0       |
| Total size               | 9.53 GB |

## Top-level Dataset Folders

| Folder                  | Files |    Size |
| ----------------------- | ----: | ------: |
| libero_10_no_noops      |    34 | 3.41 GB |
| libero_goal_no_noops    |    18 | 1.72 GB |
| libero_object_no_noops  |    34 | 2.62 GB |
| libero_spatial_no_noops |    18 | 1.78 GB |

## Episode Count Check

The local dataset contains four TFDS-style sub-datasets:

| Sub-dataset             | Split | Episodes |
| ----------------------- | ----- | -------: |
| libero_10_no_noops      | train |      379 |
| libero_goal_no_noops    | train |      428 |
| libero_object_no_noops  | train |      454 |
| libero_spatial_no_noops | train |      432 |
| Total                   | train |     1693 |

Only the `train` split was confirmed during this check.

## Step Count Planning Assumption

A small step-count sample was checked on `libero_goal_no_noops`.

Sample result:

| Item                                             |  Value |
| ------------------------------------------------ | -----: |
| sampled episodes                                 |     30 |
| sampled total steps                              |   3547 |
| average steps per sampled episode                | 118.23 |
| estimated total steps for `libero_goal_no_noops` |  50603 |

For conservative planning, this project assumes:

* 150–180 steps per episode
* approximately 250,000–300,000 total trajectory steps

This estimate is used only for planning. It should not be treated as the exact full step count.

## Check Result

Download check passed.

No `.incomplete` files were found.

The dataset contains 1693 confirmed episodes.

## Notes

This dataset is stored on the cloud data disk and should not be committed to GitHub.

Only lightweight files such as configs, logs, notes, scripts, and result tables should be committed.

The larger 124GB real-world robot dataset will be considered later as an extension stage. It should first be used as a small verified subset, not as the first full-scale experiment.
