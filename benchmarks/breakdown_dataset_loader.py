import json
import os
from pathlib import Path
from models.benchmark_dialogues import (
    DBDCDialogue,
    DBDCErrorClassificationDialogue,
)


def load_dataset(
    challenge: str = "dbdc5", split: str = "dev", lang: str = "en"
) -> list[DBDCDialogue]:
    assert lang in ["en", "ja"]
    assert split in ["dev", "eval"]
    assert challenge in ["dbdc4", "dbdc5"]
    if challenge == "dbdc4":
        assert split == "eval"
        if lang == "ja":
            lang = "jp"

        dbdc_dataset_dir = f"./datasets/dialogue_breakdowns/DBDC4_eval_20200314/{lang}/"
    elif challenge == "dbdc5":
        if lang == "ja":
            assert split == "dev"
        if lang == "en":
            dbdc_num = "4"
        elif lang == "ja":
            dbdc_num = "5"

        dbdc_dataset_dir = f"./datasets/dialogue_breakdowns/dbdc5/release-v3-distrib/dialogue_breakdown_detection/dbdc{dbdc_num}_{lang}_{split}_labeled"

    dbdc_split_dataset = []
    for root, _, files in os.walk(dbdc_dataset_dir):
        for filename in files:
            if filename.endswith(".json"):
                with open(os.path.join(root, filename), encoding="utf-8") as f:
                    sample_json = json.load(f)
                    dialogue_id = filename.split(".")[0]
                    sample = DBDCDialogue(dialogue_id=dialogue_id, **sample_json)
                    dbdc_split_dataset.append(sample)

    return dbdc_split_dataset


def load_error_classification_dataset() -> list[DBDCErrorClassificationDialogue]:
    dbdc_error_classification_dataset_dir = "./datasets/dialogue_breakdowns/dbdc5/release-v3-distrib/error_category_classification/dbdc5_ja_dev_labeled"

    dbdc_error_classification_dataset = []
    for root, _, files in os.walk(dbdc_error_classification_dataset_dir):
        for filename in files:
            if filename.endswith(".json"):
                with open(os.path.join(root, filename), encoding="utf-8") as f:
                    sample_json = json.load(f)
                    dialogue_id = filename.split(".")[0]
                    sample = DBDCErrorClassificationDialogue(
                        dialogue_id=dialogue_id, **sample_json
                    )
                    dbdc_error_classification_dataset.append(sample)

    return dbdc_error_classification_dataset


def load_tested_dialogues(
    challenge: str = "dbdc5", split: str = "dev", lang: str = "en"
) -> list[DBDCDialogue]:
    assert lang in ["en", "ja"]
    assert split in ["dev", "eval"]
    assert challenge in ["dbdc4", "dbdc5"]
    if challenge == "dbdc4":
        assert split == "eval"
    elif challenge == "dbdc5":
        if lang == "ja":
            assert split == "dev"

    eval_base_dir = Path(f"./data/{challenge}_{lang}_{split}_subset/")
    tested_subset_dir = eval_base_dir / "annotated_dialogues"

    # Load all dialogue.yaml files from the tested subset directory
    dialogue_files = [f for f in tested_subset_dir.glob("*.json")]
    tested_dialogues = []
    for dialogue_file in dialogue_files:
        with open(dialogue_file, "r", encoding="utf-8") as f:
            dialogue_dict = json.load(f)
            dialogue = DBDCDialogue(**dialogue_dict)
            tested_dialogues.append(dialogue)
    return tested_dialogues


def load_tested_error_classification_dialogues() -> (
    list[DBDCErrorClassificationDialogue]
):
    eval_base_dir = Path("./data/dbdc5_error_classification_ja_dev_subset/")
    tested_subset_dir = eval_base_dir / "annotated_dialogues"

    dialogue_files = [f for f in tested_subset_dir.glob("*.json")]
    tested_dialogues = []
    for dialogue_file in dialogue_files:
        with open(dialogue_file, "r", encoding="utf-8") as f:
            dialogue_dict = json.load(f)
            dialogue = DBDCErrorClassificationDialogue(**dialogue_dict)
            tested_dialogues.append(dialogue)
    return tested_dialogues


if __name__ == "__main__":
    # Load the dataset
    dbdc_dataset = load_dataset("dev")

    # Print the number of samples loaded
    print(f"Loaded {len(dbdc_dataset)} samples from the dataset.")
    # Print the first sample to verify
    print(dbdc_dataset[0])

    # Convert to chat_checker dialogues
    chat_checker_dialogues = [
        sample.to_chat_checker_dialogue() for sample in dbdc_dataset
    ]

    # Print the first dialogue to verify
    print(chat_checker_dialogues[0])
