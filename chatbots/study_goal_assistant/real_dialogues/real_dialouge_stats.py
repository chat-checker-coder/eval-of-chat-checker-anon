import json
from pathlib import Path
from typing import Dict
import re

from lexical_diversity import lex_div
import numpy as np

from chat_checker.utils.misc_utils import five_num_summary


def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    # Remove punctuation and split on whitespace
    words = re.findall(r"\w+", text.lower())
    return len(words)


def analyze_dialogue_turns(file_path: str) -> Dict[str, float]:
    dialogues = []
    user_turns = []
    chatbot_turns = []

    results_array = []
    with open(file_path, "r", encoding="utf-8") as f:
        results_array = json.load(f)

    for result in results_array:
        responses = result["results"]
        for response in responses:
            if not response.get("answer"):
                continue
            if not isinstance(response["answer"], dict):
                continue
            answer = response["answer"]
            if not answer.get("messageHistory"):
                continue
            if answer.get("is_stupid"):
                continue
            if not answer.get("feedback"):
                continue
            if not answer.get("cd_examples"):
                continue
            dialogue = answer["messageHistory"]
            dialogues.append(dialogue)

            for turn in dialogue:
                if turn["role"] == "MessageRole.user":
                    user_turns.append(turn)
                else:
                    chatbot_turns.append(turn)

    n_dialogues = len(dialogues)
    n_user_turns = len(user_turns)
    n_chatbot_turns = len(chatbot_turns)

    dialogue_lengths = [len(dialogue) for dialogue in dialogues]

    dialogue_length_five_num_summary = five_num_summary(dialogue_lengths)
    avg_dialogue_length = sum(dialogue_lengths) / len(dialogue_lengths)
    std_dialogue_length = np.std(dialogue_lengths)

    # Filter outliers from the dialogue lengths (based on the 1.5 * IQR rule)
    q1 = dialogue_length_five_num_summary["q1"]
    q3 = dialogue_length_five_num_summary["q3"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    dialogue_lengths = [
        length for length in dialogue_lengths if lower_bound <= length <= upper_bound
    ]
    no_outlier_dialogue_length_five_num_summary = five_num_summary(dialogue_lengths)

    if n_dialogues > 0:
        avg_user_turns_per_dialogue = n_user_turns / n_dialogues
        avg_chatbot_turns_per_dialogue = n_chatbot_turns / n_dialogues
    else:
        avg_user_turns_per_dialogue = 0.0
        avg_chatbot_turns_per_dialogue = 0.0

    user_turn_lengths = [len(turn["content"].split()) for turn in user_turns]
    chatbot_turn_lengths = [len(turn["content"].split()) for turn in chatbot_turns]

    # print the user turn with the most words
    max_user_turn_length = max(user_turn_lengths)
    max_user_turn_index = user_turn_lengths.index(max_user_turn_length)
    print(
        f'User turn with the most words: "{user_turns[max_user_turn_index]["content"]}"'
    )

    # print the user turn with the least words
    min_user_turn_length = min(user_turn_lengths)
    min_user_turn_index = user_turn_lengths.index(min_user_turn_length)
    print(
        f'User turn with the least words: "{user_turns[min_user_turn_index]["content"]}"'
    )

    if n_user_turns > 0:
        avg_user_turn_length = sum(user_turn_lengths) / n_user_turns
        std_user_turn_length = np.std(user_turn_lengths)
    else:
        avg_user_turn_length = 0.0
        std_user_turn_length = 0.0

    if n_chatbot_turns > 0:
        avg_chatbot_turn_length = sum(chatbot_turn_lengths) / n_chatbot_turns
        std_chatbot_turn_length = np.std(chatbot_turn_lengths)
    else:
        avg_chatbot_turn_length = 0.0
        std_chatbot_turn_length = 0.0

    user_turn_length_five_num_summary = five_num_summary(user_turn_lengths)
    chatbot_turn_length_five_num_summary = five_num_summary(chatbot_turn_lengths)

    user_turn_tokens = []
    for turn in user_turns:
        user_turn_tokens.extend(lex_div.tokenize(turn["content"]))

    chatbot_turn_tokens = []
    for turn in chatbot_turns:
        chatbot_turn_tokens.extend(lex_div.tokenize(turn["content"]))

    user_turn_mtld = lex_div.mtld(user_turn_tokens)
    chatbot_turn_mtld = lex_div.mtld(chatbot_turn_tokens)

    return {
        "n_dialogues": n_dialogues,
        "n_user_turns": n_user_turns,
        "n_chatbot_turns": n_chatbot_turns,
        "dialogue_length_five_num_summary": dialogue_length_five_num_summary,
        "no_outlier_dialogue_length_five_num_summary": no_outlier_dialogue_length_five_num_summary,
        "avg_dialogue_length": avg_dialogue_length,
        "std_dialogue_length": std_dialogue_length,
        "avg_user_turns_per_dialogue": avg_user_turns_per_dialogue,
        "avg_chatbot_turns_per_dialogue": avg_chatbot_turns_per_dialogue,
        "average_user_turn_length": avg_user_turn_length,
        "std_user_turn_length": std_user_turn_length,
        "average_chatbot_turn_length": avg_chatbot_turn_length,
        "std_chatbot_turn_length": std_chatbot_turn_length,
        "user_turn_length_five_num_summary": user_turn_length_five_num_summary,
        "chatbot_turn_length_five_num_summary": chatbot_turn_length_five_num_summary,
        "user_turn_mtld": user_turn_mtld,
        "chatbot_turn_mtld": chatbot_turn_mtld,
    }


def main():
    # Get the path to the study responses file
    current_dir = Path(__file__).parent
    study_responses_path = current_dir / "study_responses.json"

    # Analyze the dialogues
    stats = analyze_dialogue_turns(str(study_responses_path))

    # Print the results
    print("\nDialogue Statistics:")
    print(f"Number of dialogues: {stats['n_dialogues']}")
    print(f"Number of user turns: {stats['n_user_turns']}")
    print(f"Number of chatbot turns: {stats['n_chatbot_turns']}")
    print(
        f"Dialogue turns five number summary: {stats['dialogue_length_five_num_summary']}"
    )
    print(
        f"Dialogue turns five number summary (no outliers): {stats['no_outlier_dialogue_length_five_num_summary']}"
    )
    print(f"Average dialogue length: {stats['avg_dialogue_length']:.2f} turns")
    print(
        f"Standard deviation of dialogue length: {stats['std_dialogue_length']:.2f} turns"
    )
    print(
        f"Average user turns per dialogue: {stats['avg_user_turns_per_dialogue']:.2f}"
    )
    print(
        f"Average chatbot turns per dialogue: {stats['avg_chatbot_turns_per_dialogue']:.2f}"
    )
    print(f"Average user turn length: {stats['average_user_turn_length']:.2f} words")
    print(
        f"Standard deviation of user turn length: {stats['std_user_turn_length']:.2f} words"
    )
    print(
        f"Average chatbot turn length: {stats['average_chatbot_turn_length']:.2f} words"
    )
    print(
        f"Standard deviation of chatbot turn length: {stats['std_chatbot_turn_length']:.2f} words"
    )
    print(
        f"User turn length five number summary: {stats['user_turn_length_five_num_summary']}"
    )
    print(
        f"Chatbot turn length five number summary: {stats['chatbot_turn_length_five_num_summary']}"
    )
    print(f"User turn MTLD: {stats['user_turn_mtld']:.2f}")
    print(f"Chatbot turn MTLD: {stats['chatbot_turn_mtld']:.2f}")


if __name__ == "__main__":
    main()
