OpenVLA real BridgeData V2 sample image package

Purpose:
- These 10 real robot-manipulation images are for quick OpenVLA cloud smoke testing.
- They are taken from the public BridgeData V2 project page teaser images.
- They are not a full training dataset and should not be used as the final dissertation evaluation set.
- Use them to verify image_path reading, instruction input, model loading, sample action output, latency, and VRAM logging.

Suggested cloud usage:
unzip openvla_real_bridge_sample_images.zip
python scripts/run_openvla_baseline.py --image-path data/sample_images/test_02_dishwasher_cups.jpg --instruction "pick up any cup"

Files:
- data/sample_images/*.jpg
- test_instructions.csv
- IMAGE_SOURCES.txt
