import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to dataset directory")
    args = parser.parse_args()

    DATASET_PATH = Path(args.data_dir)

    # "train" or "test"
    for stage in ["test"]:
        stage = DATASET_PATH / stage

        index = {}
        for chunk_path in tqdm(
            sorted(list(stage.iterdir())), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)