import argparse
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path)
args = parser.parse_args()

random.seed(42)

train = args.data_dir / "train"
val = args.data_dir / "val"

for class_dir in sorted(train.iterdir()):
    # Error out if we already have the val split
    (val / class_dir.name).mkdir(parents=True, exist_ok=False)
    images = sorted(class_dir.iterdir())
    chosen = random.sample(images, len(images) // 10)

    for img in chosen:
        img.rename(val / class_dir.name / img.name)
