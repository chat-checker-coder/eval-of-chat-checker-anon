from pathlib import Path
import json

from chat_checker.models.dialogue import DialogueTurn, SpeakerRole

from chat_checker.models.rating import RatingDimensionAnnotation, RatingScale
from models.benchmark_dialogues import (
    EvaluatedDialogue,
)

BASE_DIR = Path(__file__).parent
TASK_ORIENTED_DIALOGUES_DIR = (
    BASE_DIR / "datasets/task_oriented_dialogue_systems/"
).resolve()
CONVERSATIONAL_DIALOGUES_DIR = (
    BASE_DIR / "datasets/conversational_dialogue_systems/"
).resolve()


def get_uss_dataset(name="SGD") -> list[EvaluatedDialogue]:
    dialogues: list[EvaluatedDialogue] = []
    dialogue_id = 1

    with open(
        f"{TASK_ORIENTED_DIALOGUES_DIR.as_posix()}/uss/{name}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        dialogue = None

        for i, line in enumerate(f):
            line = line.strip()

            if not line:
                # End of a dialogue
                if dialogue:
                    dialogues.append(dialogue)

                    # Reset variables for the next dialogue
                    dialogue = None
                    dialogue_id += 1
                continue

            # Start a new dialogue if none is active
            if dialogue is None:
                dialogue = EvaluatedDialogue(
                    dialogue_id=str(dialogue_id), chat_history=[], rating_annotations={}
                )

            # Split the line into fields
            fields = line.split("\t")

            if len(fields) >= 4:
                # User utterances
                role = fields[0].strip()
                text = fields[1].strip()
                #  action = fields[2].strip()
                satisfaction = fields[3].strip()
            elif len(fields) == 3:
                # System utterances
                role = fields[0].strip()
                text = fields[1].strip()
                # action = fields[2].strip()
                satisfaction = ""
            elif len(fields) == 2:
                # System utterances without action
                role = fields[0].strip()
                text = fields[1].strip()
                satisfaction = ""
            else:
                # Handle cases with insufficient fields
                print(f"Warning: Insufficient fields ({fields}) in line {i}. Skipping.")
                continue

            if text == "OVERALL":
                # Last line of the conversation with the overall satisfaction rating
                ratings = [
                    int(r) for r in satisfaction.split(",") if r.strip().isdigit()
                ]
                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                else:
                    avg_rating = None

                dialogue.rating_annotations["overall"] = RatingDimensionAnnotation(
                    ratings=ratings, avg_rating=avg_rating
                )
            else:
                # Normal line of the conversation
                # Parse satisfaction ratings
                ratings = [
                    int(r) for r in satisfaction.split(",") if r.strip().isdigit()
                ]
                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                else:
                    avg_rating = None

                # Normalize role to 'user' or 'dialogue_system'
                role_lower = role.lower()
                if role_lower == "user":
                    role_normalized = SpeakerRole.USER
                elif role_lower == "system":
                    role_normalized = SpeakerRole.DIALOGUE_SYSTEM
                else:
                    raise ValueError(f"Unknown role '{role}' in line {i}.")

                # Create a dialogue turn dictionary
                turn = DialogueTurn(
                    turn_id=len(dialogue.chat_history) + 1,
                    role=role_normalized,
                    content=text,
                )
                dialogue.chat_history.append(turn)

        # Process the last dialogue if the file doesn't end with a blank line
        if dialogue:
            dialogues.append(dialogue)

    return dialogues


def get_fed_dial_dataset() -> list[EvaluatedDialogue]:
    with open(
        f"{CONVERSATIONAL_DIALOGUES_DIR}/dstc_10_track_5/Subtask 1/human_evaluation_data/fed-dial_eval.json",
        "r",
        encoding="utf-8",
    ) as f:
        raw_dialogues = json.load(f)

    loaded_dialogues: list[EvaluatedDialogue] = []

    original_scales = {
        "coherent": [0, 2],
        "error_recovery": [0, 2],
        "consistent": [0, 1],
        "diverse": [0, 2],
        "depth": [0, 2],
        "likeable": [0, 2],
        "understanding": [0, 2],
        "flexible": [0, 2],
        "informative": [0, 2],
        "inquisitive": [0, 2],
        "overall": [0, 4],
    }

    role_mapping = {"human": SpeakerRole.USER, "model": SpeakerRole.DIALOGUE_SYSTEM}

    for item in raw_dialogues:
        dialogue = EvaluatedDialogue(
            dialogue_id=item.get("dialogue_id", ""),
            chat_history=[],
            rating_annotations={},
        )

        for turn in item.get("dialog", []):
            speaker = turn.get("speaker", "")
            role = role_mapping.get(speaker, speaker)
            turn = DialogueTurn(
                turn_id=len(dialogue.chat_history) + 1,
                role=role,
                content=turn.get("text", ""),
            )
            dialogue.chat_history.append(turn)

        # Set model
        dialogue.chat_model = item.get("model", "")

        # Transform annotations
        for key, ratings in item.get("annotations", {}).items():
            # Map the key to lower case and replace spaces with underscores
            new_key = key.lower().replace(" ", "_")
            scale = original_scales.get(
                new_key, [0, 2]
            )  # Default scale [0, 2] if not specified
            # Move rating scale by +1 to match the 1-x scale we use
            ratings = [r + 1 for r in ratings] if ratings else []
            scale = [s + 1 for s in scale]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            rating_annotation = RatingDimensionAnnotation(
                ratings=ratings,
                avg_rating=avg_rating,
                scale=RatingScale(min=scale[0], max=scale[1]),
            )
            dialogue.rating_annotations[new_key] = rating_annotation

        loaded_dialogues.append(dialogue)

    return loaded_dialogues


if __name__ == "__main__":
    uss_dataset = "MWOZ"
    dialogues = get_uss_dataset(uss_dataset)
    print(f"Loaded {len(dialogues)} dialogues from the {uss_dataset} dataset.")
    print(json.dumps(dialogues[0].model_dump(), indent=4))
    print(json.dumps(dialogues[1].model_dump(), indent=4))
    print(json.dumps(dialogues[-1].model_dump(), indent=4))

    # Write the dialogues to a JSON file
    with open(
        f"{TASK_ORIENTED_DIALOGUES_DIR}/uss/{uss_dataset}.json", "w+", encoding="utf-8"
    ) as f:
        json.dump(dialogues, f, indent=4, ensure_ascii=False)

    # dialogues = get_fed_dial_dataset()
    # print(f"Loaded {len(dialogues)} dialogues from the FedDial dataset.")
    # print(json.dumps(dialogues[0].model_dump(), indent=4))
    # print(json.dumps(dialogues[1].model_dump(), indent=4))
    # print(json.dumps(dialogues[-1].model_dump(), indent=4))

    # # Write the dialogues to a JSON file
    # with open(
    #     f"{CONVERSATIONAL_DIALOGUES_DIR}/dstc_10_track_5/Subtask 1/human_evaluation_data/fed_dial_formatted.json",
    #     "w+",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump([d.model_dump() for d in dialogues], f, indent=4, ensure_ascii=False)
