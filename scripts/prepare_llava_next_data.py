import json
import os

from datasets import load_dataset
from tqdm import tqdm


BASE_DIR = "/leonardo_scratch/large/userexternal/fmohamma/zsc/llava_data"
IMAGE_DIR = os.path.join(BASE_DIR, "llava_1_6_images")
JSON_PATH = os.path.join(BASE_DIR, "llava_1_6.json")
NUM_PROC = 8


def main() -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)

    data = load_dataset(
        "lmms-lab/LLaVA-NeXT-Data",
        split="train",
        cache_dir=BASE_DIR,
    )

    def process_example(example):
        result = {
            "id": example["id"],
            "conversations": example["conversations"],
        }
        if example["image"] is not None:
            image_name = f"{example['id']}.jpg"
            result["image"] = image_name
            example["image"].save(os.path.join(IMAGE_DIR, image_name))
        return result

    processed = data.map(
        process_example,
        num_proc=NUM_PROC,
        remove_columns=data.column_names,
        desc="Saving images",
    )

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        file.write("[\n")
        for idx, item in enumerate(tqdm(processed, desc="Writing JSON")):
            if idx:
                file.write(",\n")
            json.dump(item, file, ensure_ascii=False)
        file.write("\n]\n")


if __name__ == "__main__":
    main()
