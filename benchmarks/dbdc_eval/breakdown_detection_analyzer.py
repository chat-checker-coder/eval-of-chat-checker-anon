from pathlib import Path
from models.benchmark_dialogues import (
    DBDCDialogue,
    DBDCErrorClassificationDialogue,
    DBDCSpeakerRole,
)
from collections import Counter
from typing import Any, Dict
from dbdc_eval.breakdown_detection_evaluator import (
    calc_distribution,
    compute_dbdc_scores,
    majority_label,
    majority_label_lenient,
)


def analyze_dataset(dataset: list[DBDCDialogue]) -> Dict[str, Any]:
    """
    Analyze the dataset and compute various statistics.

    Args:
        dataset: List of DBDCDialogue objects

    Returns:
        Dictionary containing:
        - num_dialogues: Total number of dialogues
        - num_system_turns: Total number of system turns
        - avg_system_turn_length: Average number of words in system turns
        - strict_label_counts: Counter of labels (O, T, X) using strict labeling
        - lenient_label_counts: Counter of labels (O, X) using lenient labeling
    """
    num_dialogues = len(dataset)
    num_system_turns = 0
    total_system_words = 0
    strict_label_counts: Counter = Counter()
    lenient_label_counts: Counter = Counter()

    for dialogue in dataset:
        for turn in dialogue.turns:
            if turn.speaker != DBDCSpeakerRole.SYSTEM or not turn.annotations:
                continue

            num_system_turns += 1
            total_system_words += len(turn.utterance.split())

            # Calculate distribution and get majority labels
            prob_dist = calc_distribution(turn.annotations)
            strict_label = majority_label(
                prob_dist[0], prob_dist[1], prob_dist[2], threshold=0.0
            )
            lenient_label = majority_label_lenient(
                prob_dist[0], prob_dist[1], prob_dist[2], threshold=0.0
            )

            # Update counters
            strict_label_counts[strict_label] += 1
            lenient_label_counts[lenient_label] += 1

    # Calculate percentages for strict labels
    strict_label_percentages = {
        label: (count / num_system_turns * 100) if num_system_turns > 0 else 0
        for label, count in strict_label_counts.items()
    }

    # Calculate percentages for lenient labels
    lenient_label_percentages = {
        label: (count / num_system_turns * 100) if num_system_turns > 0 else 0
        for label, count in lenient_label_counts.items()
    }

    return {
        "num_dialogues": num_dialogues,
        "num_system_turns": num_system_turns,
        "avg_system_turn_per_dialogue": num_system_turns / num_dialogues,
        "avg_system_turn_length": total_system_words / num_system_turns
        if num_system_turns > 0
        else 0,
        "strict_label_counts": dict(strict_label_counts),
        "lenient_label_counts": dict(lenient_label_counts),
        "strict_label_percentages": strict_label_percentages,
        "lenient_label_percentages": lenient_label_percentages,
    }


def analyze_configs(
    tested_samples: list[DBDCDialogue],
    config_keys: list[str],
    eval_dir: Path,
):
    results = {}
    for config_key in config_keys:
        print(f"Analyzing {config_key}...")
        res = analyze_config_results(tested_samples, config_key, eval_dir)
        res.print_results()
        results[config_key] = res

    return results


def analyze_config_results(
    tested_samples: list[DBDCDialogue], config_key: str, eval_dir: Path
):
    res = compute_dbdc_scores(tested_samples, config_key)

    # Save results to file
    with open(eval_dir / f"{config_key}.txt", "w") as f:
        f.write(str(res))

    print(f"Saved results to {eval_dir / f'{config_key}.txt'}")

    return res


def analyze_error_category_classification_dataset(
    dataset: list[DBDCErrorClassificationDialogue],
) -> Dict[str, Any]:
    """
    Analyze the dataset and compute various statistics.

    Args:
        dataset: List of DBDCErrorClassificationDialogue objects

    Returns:
        Dictionary containing:
        - num_dialogues: Total number of dialogues
        - num_system_turns: Total number of system turns
        - avg_system_turn_length: Average number of words in system turns
        - strict_label_counts: Counter of labels (O, T, X) using strict labeling
        - lenient_label_counts: Counter of labels (O, X) using lenient labeling
        - error_category_counts: Counter of error categories
    """
    num_dialogues = len(dataset)
    num_system_turns = 0
    total_system_words = 0
    strict_label_counts: Counter = Counter()
    lenient_label_counts: Counter = Counter()
    error_category_counts: Counter = Counter()

    for dialogue in dataset:
        for turn in dialogue.turns:
            if turn.speaker != DBDCSpeakerRole.SYSTEM:
                continue

            num_system_turns += 1
            total_system_words += len(turn.utterance.split())

            # Calculate distribution and get majority labels
            prob_dist = [turn.num_O, turn.num_T, turn.num_X]
            strict_label = majority_label(
                prob_dist[0], prob_dist[1], prob_dist[2], threshold=0.0
            )
            lenient_label = majority_label_lenient(
                prob_dist[0], prob_dist[1], prob_dist[2], threshold=0.0
            )

            # Update counters
            strict_label_counts[strict_label] += 1
            lenient_label_counts[lenient_label] += 1

            # Count error categories if present
            if turn.error_category:
                for category in turn.error_category:
                    error_category_counts[category] += 1

    # Calculate percentages for strict labels
    strict_label_percentages = {
        label: (count / num_system_turns * 100) if num_system_turns > 0 else 0
        for label, count in strict_label_counts.items()
    }

    # Calculate percentages for lenient labels
    lenient_label_percentages = {
        label: (count / num_system_turns * 100) if num_system_turns > 0 else 0
        for label, count in lenient_label_counts.items()
    }

    # Calculate percentages for error categories
    error_category_percentages = {
        category: (count / num_system_turns * 100) if num_system_turns > 0 else 0
        for category, count in error_category_counts.items()
    }

    return {
        "num_dialogues": num_dialogues,
        "num_system_turns": num_system_turns,
        "avg_system_turn_per_dialogue": num_system_turns / num_dialogues,
        "avg_system_turn_length": total_system_words / num_system_turns
        if num_system_turns > 0
        else 0,
        "strict_label_counts": dict(strict_label_counts),
        "lenient_label_counts": dict(lenient_label_counts),
        "strict_label_percentages": strict_label_percentages,
        "lenient_label_percentages": lenient_label_percentages,
        "error_category_counts": dict(error_category_counts),
        "error_category_percentages": error_category_percentages,
    }
