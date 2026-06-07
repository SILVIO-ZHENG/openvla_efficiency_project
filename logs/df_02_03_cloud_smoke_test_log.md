# DF-02-03 Cloud Baseline Smoke Test Batch Run

## Timestamp
- 2026-06-07 23:21:07

## Git Information
- Branch: `DF-02-03`
- Commit: `ad3cc72`

## Configuration
- Model ID: `openvla/openvla-7b`
- Device: `cuda:0`
- Dtype: `bfloat16`
- Instruction CSV: `data\sample_images\test_instructions.csv`
- Warm-up runs per sample: `1`
- Measured runs per sample: `1`
- Sample limit: `10`
- Dry run: `True`

## Sample List
- `sample_001` | `data/sample_images/test_01_laundry_basket.jpg` | exists=`True` | instruction=`put clothes into the laundry machine`
- `sample_002` | `data/sample_images/test_02_dishwasher_cups.jpg` | exists=`True` | instruction=`pick up any cup`
- `sample_003` | `data/sample_images/test_03_tool_chest.jpg` | exists=`True` | instruction=`pick up the closest Allen key set`
- `sample_004` | `data/sample_images/test_04_tabletop_spoon.jpg` | exists=`True` | instruction=`pick up the spoon`
- `sample_005` | `data/sample_images/test_05_beet_pot_sink.jpg` | exists=`True` | instruction=`put the beet in the pot`
- `sample_006` | `data/sample_images/test_06_banana_pot.jpg` | exists=`True` | instruction=`put the banana in the pot`
- `sample_007` | `data/sample_images/test_07_blueberry_plate.jpg` | exists=`True` | instruction=`put the blueberry on the plate`
- `sample_008` | `data/sample_images/test_08_sponge_plate.jpg` | exists=`True` | instruction=`pick up the sponge and wipe the plate`
- `sample_009` | `data/sample_images/test_09_set_table.jpg` | exists=`True` | instruction=`set the table`
- `sample_010` | `data/sample_images/test_10_knife_cutting_board.jpg` | exists=`True` | instruction=`put the knife on the cutting board`

## Dry Run Result
- No model was loaded.
- CSV reading succeeded.
- Image path visibility was checked.


