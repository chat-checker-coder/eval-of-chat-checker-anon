#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Adapted from https://github.com/dbd-challenge/dbdc4/tree/aecf252505e9b1736723848b78eb55aaf5712b4d/prog/eval

import math
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
from pydantic import BaseModel

from chat_checker.models.breakdowns import BreakdownDecision
from chat_checker.breakdown_detection.breakdown_taxonomy import get_breakdown_title_list
from models.benchmark_dialogues import (
    DBDCErrorClassificationDialogue,
    DBDCErrorClassificationTurn,
    DBDCBreakdownCategory,
    DBDCSpeakerRole,
)

plt.style.use("seaborn-v0_8")
# set tick and label font size
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["legend.fontsize"] = "large"


class ClassificationMetrics(BaseModel):
    accuracy: float
    correct_count: int
    incorrect_count: int
    precision: float
    recall: float
    f_measure: float
    true_positives: int
    false_positives: int
    false_negatives: int


class DistributionMetrics(BaseModel):
    jsd_O_T_X: float
    jsd_O_TX: float
    jsd_OT_X: float
    mse_O_T_X: float
    mse_O_TX: float
    mse_OT_X: float


class DBDCEvaluationResults(BaseModel):
    file_count: int
    label_count: int

    # Label distribution
    o_label_count: int
    t_label_count: int
    x_label_count: int

    # Strict evaluation metrics
    strict_metrics: ClassificationMetrics

    # Lenient evaluation metrics
    lenient_metrics: ClassificationMetrics

    # Distribution metrics
    distribution_metrics: DistributionMetrics

    def __str__(self) -> str:
        lines = []
        lines.append("######### Data Stats #########")
        lines.append("File Num : \t\t" + str(self.file_count))
        lines.append("System Utterance Num : \t" + str(self.label_count))
        lines.append("O Label Num : \t\t" + str(self.o_label_count))
        lines.append("T Label Num : \t\t" + str(self.t_label_count))
        lines.append("X Label Num : \t\t" + str(self.x_label_count))
        lines.append("")

        lines.append("######### Results #########")
        lines.append(
            "Accuracy : \t\t%4f" % self.strict_metrics.accuracy
            + " ("
            + str(self.strict_metrics.correct_count)
            + "/"
            + str(
                self.strict_metrics.correct_count + self.strict_metrics.incorrect_count
            )
            + ")\n"
        )

        lines.append(
            "Lenient Accuracy : \t%4f" % self.lenient_metrics.accuracy
            + " ("
            + str(self.lenient_metrics.correct_count)
            + "/"
            + str(
                self.lenient_metrics.correct_count
                + self.lenient_metrics.incorrect_count
            )
            + ")\n"
        )

        lines.append(
            "Precision (X) : \t%4f" % self.strict_metrics.precision
            + " ("
            + str(self.strict_metrics.true_positives)
            + "/"
            + str(
                self.strict_metrics.true_positives + self.strict_metrics.false_positives
            )
            + ")"
        )
        lines.append(
            "Recall    (X) : \t%4f" % self.strict_metrics.recall
            + " ("
            + str(self.strict_metrics.true_positives)
            + "/"
            + str(
                self.strict_metrics.true_positives + self.strict_metrics.false_negatives
            )
            + ")"
        )
        lines.append("F-measure (X) : \t%4f" % self.strict_metrics.f_measure + "\n")

        lines.append(
            "Precision (T+X) : \t%4f" % self.lenient_metrics.precision
            + " ("
            + str(self.lenient_metrics.true_positives)
            + "/"
            + str(
                self.lenient_metrics.true_positives
                + self.lenient_metrics.false_positives
            )
            + ")"
        )
        lines.append(
            "Recall    (T+X) : \t%4f" % self.lenient_metrics.recall
            + " ("
            + str(self.lenient_metrics.true_positives)
            + "/"
            + str(
                self.lenient_metrics.true_positives
                + self.lenient_metrics.false_negatives
            )
            + ")"
        )
        lines.append("F-measure (T+X) : \t%4f" % self.lenient_metrics.f_measure + "\n")

        lines.append(
            "JS divergence (O,T,X) : \t%4f" % self.distribution_metrics.jsd_O_T_X
        )
        lines.append(
            "JS divergence (O,T+X) : \t%4f" % self.distribution_metrics.jsd_O_TX
        )
        lines.append(
            "JS divergence (O+T,X) : \t%4f" % self.distribution_metrics.jsd_OT_X + "\n"
        )

        lines.append(
            "Mean squared error (O,T,X) : \t%4f" % self.distribution_metrics.mse_O_T_X
        )
        lines.append(
            "Mean squared error (O,T+X) : \t%4f" % self.distribution_metrics.mse_O_TX
        )
        lines.append(
            "Mean squared error (O+T,X) : \t%4f" % self.distribution_metrics.mse_OT_X
        )
        lines.append("###########################")

        return "\n".join(lines)

    def print_results(self):
        print(str(self))


def calc_distribution(turn: DBDCErrorClassificationTurn) -> List[float]:
    num_annotations = turn.num_annotations

    prob_O = turn.num_O / num_annotations
    prob_T = turn.num_T / num_annotations
    prob_X = turn.num_X / num_annotations

    return [prob_O, prob_T, prob_X]


def majority_label(
    prob_O: float, prob_T: float, prob_X: float, threshold: float
) -> DBDCBreakdownCategory:
    if prob_O >= prob_T and prob_O >= prob_X and prob_O >= threshold:
        return DBDCBreakdownCategory.NO_BREAKDOWN
    elif prob_T >= prob_O and prob_T >= prob_X and prob_T >= threshold:
        return DBDCBreakdownCategory.POSSIBLE_BREAKDOWN
    elif prob_X >= prob_T and prob_X >= prob_O and prob_X >= threshold:
        return DBDCBreakdownCategory.BREAKDOWN
    else:
        return DBDCBreakdownCategory.NO_BREAKDOWN


def majority_label_lenient(
    prob_O: float, prob_T: float, prob_X: float, threshold: float
) -> str:
    if prob_O >= prob_T + prob_X and prob_O >= threshold:
        return "O_l"
    elif prob_T + prob_X >= prob_O and prob_T + prob_X >= threshold:
        return "X_l"
    else:
        return "O_l"


def kld(p: List[float], q: List[float]) -> float:
    k = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            k += p[i] * (math.log(p[i] / q[i], 2))
    return k


def jsd(p: List[float], q: List[float]) -> float:
    m = [(p[i] + q[i]) / 2.0 for i in range(len(p))]
    return (kld(p, m) + kld(q, m)) / 2.0


def mse(p: List[float], q: List[float]) -> float:
    total = sum(pow(p[i] - q[i], 2) for i in range(len(p)))
    return total / len(p)


def compute_dbdc_detection_scores(
    dialogues: List[DBDCErrorClassificationDialogue],
    config_key: str,
    threshold: float = 0.0,
) -> DBDCEvaluationResults:
    # Initialize counters
    file_count = 0
    label_count = 0
    strict_correct_count = 0
    strict_incorrect_count = 0
    lenient_correct_count = 0
    lenient_incorrect_count = 0

    # Strict evaluation counts
    predO_ansO = 0
    predO_ansT = 0
    predO_ansX = 0
    predT_ansO = 0
    predT_ansT = 0
    predT_ansX = 0
    predX_ansO = 0
    predX_ansT = 0
    predX_ansX = 0

    # Lenient evaluation counts
    predO_ansO_l = 0
    predO_ansX_l = 0
    predT_ansO_l = 0
    predT_ansX_l = 0
    predX_ansO_l = 0
    predX_ansX_l = 0

    # Distribution metrics
    jsd_O_T_X_sum = 0.0
    jsd_O_TX_sum = 0.0
    jsd_OT_X_sum = 0.0
    mse_O_T_X_sum = 0.0
    mse_O_TX_sum = 0.0
    mse_OT_X_sum = 0.0

    for dialogue in dialogues:
        try:
            prediction_dialogue = dialogue.to_eval_prediction_dialogue(config_key)
        except ValueError:
            # Skip dialogues that don't have the requested LLM key
            # print(
            #     f"Skipping dialogue {dialogue.dialogue_id} because it doesn't have the requested LLM key for all turns"
            # )
            continue

        label_index = 0
        file_count += 1

        for turn in dialogue.turns:
            if turn.speaker != DBDCSpeakerRole.SYSTEM or turn.num_annotations == 0:
                continue

            label_count += 1

            ans_prob_dist = calc_distribution(turn)
            ans_label = majority_label(
                ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold
            )
            ans_label_l = majority_label_lenient(
                ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold
            )

            if label_index >= len(prediction_dialogue.turns):
                continue

            pred_turn = prediction_dialogue.turns[label_index]
            pred = pred_turn.labels[0]

            pred_prob_dist = [
                float(pred.prob_O),
                float(pred.prob_T),
                float(pred.prob_X),
            ]

            # Calculate distribution metrics
            jsd_O_T_X_sum += jsd(ans_prob_dist, pred_prob_dist)
            jsd_O_TX_sum += jsd(
                [ans_prob_dist[0], ans_prob_dist[1] + ans_prob_dist[2]],
                [pred_prob_dist[0], pred_prob_dist[1] + pred_prob_dist[2]],
            )
            jsd_OT_X_sum += jsd(
                [ans_prob_dist[0] + ans_prob_dist[1], ans_prob_dist[2]],
                [pred_prob_dist[0] + pred_prob_dist[1], pred_prob_dist[2]],
            )

            mse_O_T_X_sum += mse(ans_prob_dist, pred_prob_dist)
            mse_O_TX_sum += mse(
                [ans_prob_dist[0], ans_prob_dist[1] + ans_prob_dist[2]],
                [pred_prob_dist[0], pred_prob_dist[1] + pred_prob_dist[2]],
            )
            mse_OT_X_sum += mse(
                [ans_prob_dist[0] + ans_prob_dist[1], ans_prob_dist[2]],
                [pred_prob_dist[0] + pred_prob_dist[1], pred_prob_dist[2]],
            )

            pred_label = pred.breakdown
            label_index += 1

            # Update strict evaluation counts
            if pred_label == ans_label:
                strict_correct_count += 1
            else:
                strict_incorrect_count += 1

            if pred_label == DBDCBreakdownCategory.NO_BREAKDOWN:
                if ans_label == DBDCBreakdownCategory.NO_BREAKDOWN:
                    predO_ansO += 1
                elif ans_label == DBDCBreakdownCategory.POSSIBLE_BREAKDOWN:
                    predO_ansT += 1
                elif ans_label == DBDCBreakdownCategory.BREAKDOWN:
                    predO_ansX += 1
            elif pred_label == DBDCBreakdownCategory.POSSIBLE_BREAKDOWN:
                if ans_label == DBDCBreakdownCategory.NO_BREAKDOWN:
                    predT_ansO += 1
                elif ans_label == DBDCBreakdownCategory.POSSIBLE_BREAKDOWN:
                    predT_ansT += 1
                elif ans_label == DBDCBreakdownCategory.BREAKDOWN:
                    predT_ansX += 1
            elif pred_label == DBDCBreakdownCategory.BREAKDOWN:
                if ans_label == DBDCBreakdownCategory.NO_BREAKDOWN:
                    predX_ansO += 1
                elif ans_label == DBDCBreakdownCategory.POSSIBLE_BREAKDOWN:
                    predX_ansT += 1
                elif ans_label == DBDCBreakdownCategory.BREAKDOWN:
                    predX_ansX += 1

            # Update lenient evaluation counts
            if pred_label == DBDCBreakdownCategory.NO_BREAKDOWN:
                if ans_label_l == "O_l":
                    predO_ansO_l += 1
                    lenient_correct_count += 1
                elif ans_label_l == "X_l":
                    predO_ansX_l += 1
                    lenient_incorrect_count += 1
            elif pred_label == DBDCBreakdownCategory.POSSIBLE_BREAKDOWN:
                if ans_label_l == "O_l":
                    predT_ansO_l += 1
                    lenient_incorrect_count += 1
                elif ans_label_l == "X_l":
                    predT_ansX_l += 1
                    lenient_correct_count += 1
            elif pred_label == DBDCBreakdownCategory.BREAKDOWN:
                if ans_label_l == "O_l":
                    predX_ansO_l += 1
                    lenient_incorrect_count += 1
                elif ans_label_l == "X_l":
                    predX_ansX_l += 1
                    lenient_correct_count += 1

    # Calculate metrics
    total_count = strict_correct_count + strict_incorrect_count
    strict_accuracy = strict_correct_count / total_count if total_count > 0 else 0.0
    lenient_accuracy = lenient_correct_count / total_count if total_count > 0 else 0.0

    # Strict metrics
    strict_true_positives = predX_ansX
    strict_false_positives = predX_ansO + predX_ansT
    strict_false_negatives = predO_ansX + predT_ansX

    strict_precision = (
        strict_true_positives / (strict_true_positives + strict_false_positives)
        if (strict_true_positives + strict_false_positives) > 0
        else 0.0
    )
    strict_recall = (
        strict_true_positives / (strict_true_positives + strict_false_negatives)
        if (strict_true_positives + strict_false_negatives) > 0
        else 0.0
    )
    strict_f_measure = (
        2 * strict_precision * strict_recall / (strict_precision + strict_recall)
        if (strict_precision + strict_recall) > 0
        else 0.0
    )

    # Lenient metrics
    lenient_true_positives = predT_ansX_l + predX_ansX_l
    lenient_false_positives = predT_ansO_l + predX_ansO_l
    lenient_false_negatives = predO_ansX_l

    lenient_precision = (
        lenient_true_positives / (lenient_true_positives + lenient_false_positives)
        if (lenient_true_positives + lenient_false_positives) > 0
        else 0.0
    )
    lenient_recall = (
        lenient_true_positives / (lenient_true_positives + lenient_false_negatives)
        if (lenient_true_positives + lenient_false_negatives) > 0
        else 0.0
    )
    lenient_f_measure = (
        2 * lenient_precision * lenient_recall / (lenient_precision + lenient_recall)
        if (lenient_precision + lenient_recall) > 0
        else 0.0
    )

    # Create results object
    return DBDCEvaluationResults(
        file_count=file_count,
        label_count=label_count,
        o_label_count=predO_ansO + predT_ansO + predX_ansO,
        t_label_count=predO_ansT + predT_ansT + predX_ansT,
        x_label_count=predO_ansX + predT_ansX + predX_ansX,
        strict_metrics=ClassificationMetrics(
            accuracy=strict_accuracy,
            correct_count=strict_correct_count,
            incorrect_count=strict_incorrect_count,
            precision=strict_precision,
            recall=strict_recall,
            f_measure=strict_f_measure,
            true_positives=strict_true_positives,
            false_positives=strict_false_positives,
            false_negatives=strict_false_negatives,
        ),
        lenient_metrics=ClassificationMetrics(
            accuracy=lenient_accuracy,
            correct_count=lenient_correct_count,
            incorrect_count=lenient_incorrect_count,
            precision=lenient_precision,
            recall=lenient_recall,
            f_measure=lenient_f_measure,
            true_positives=lenient_true_positives,
            false_positives=lenient_false_positives,
            false_negatives=lenient_false_negatives,
        ),
        distribution_metrics=DistributionMetrics(
            jsd_O_T_X=jsd_O_T_X_sum / total_count if total_count > 0 else 0.0,
            jsd_O_TX=jsd_O_TX_sum / total_count if total_count > 0 else 0.0,
            jsd_OT_X=jsd_OT_X_sum / total_count if total_count > 0 else 0.0,
            mse_O_T_X=mse_O_T_X_sum / total_count if total_count > 0 else 0.0,
            mse_O_TX=mse_O_TX_sum / total_count if total_count > 0 else 0.0,
            mse_OT_X=mse_OT_X_sum / total_count if total_count > 0 else 0.0,
        ),
    )


class MismatchMetrics(BaseModel):
    set_mismatch_counts: dict[frozenset[str], dict[frozenset[str], int]]
    single_mismatch_counts: dict[str, dict[str, int]]

    def __str__(self) -> str:
        lines = []
        lines.append("######### Mismatch Metrics #########")
        if self.set_mismatch_counts:
            lines.append("\nSet Mismatches (Ground Truth -> Predicted):")
            for ref_cats, pred_dict in sorted(self.set_mismatch_counts.items()):
                for pred_cats, count in sorted(pred_dict.items()):
                    lines.append(
                        f"{', '.join(sorted(ref_cats))} -> {', '.join(sorted(pred_cats))}: {count}"
                    )
        if self.single_mismatch_counts:
            lines.append("\nSingle Mismatches (Ground Truth -> Predicted):")
            for ref_cat, preds in sorted(self.single_mismatch_counts.items()):
                for pred_cat, count in sorted(preds.items()):
                    lines.append(f"{ref_cat} -> {pred_cat}: {count}")
        lines.append("###########################")
        return "\n".join(lines)


class ErrorClassificationMetrics(BaseModel):
    exact_match: float
    superset_match: float
    partial_match: float
    f1_score: float
    precision: float
    recall: float
    total_turns: int
    correct_turns: int
    correct_superset_turns: int
    partial_match_turns: int
    predicted_n_types_per_turn: float
    ground_truth_n_types_per_turn: float

    def __str__(self) -> str:
        lines = []
        lines.append("######### Results #########")
        lines.append(f"Exact Match: {self.exact_match:.3f}")
        lines.append(f"Superset Match: {self.superset_match:.3f}")
        lines.append(f"Partial Match: {self.partial_match:.3f}")
        lines.append(f"Precision: {self.precision:.3f}")
        lines.append(f"Recall: {self.recall:.3f}")
        lines.append(f"F1 Score: {self.f1_score:.3f}")
        lines.append(f"Total Turns: {self.total_turns}")
        lines.append(f"Correct Turns: {self.correct_turns}")
        lines.append(f"Correct Superset Turns: {self.correct_superset_turns}")
        lines.append(f"Partial Match Turns: {self.partial_match_turns}")
        lines.append(
            f"Predicted N Types Per Turn: {self.predicted_n_types_per_turn:.3f}"
        )
        lines.append(
            f"Ground Truth N Types Per Turn: {self.ground_truth_n_types_per_turn:.3f}"
        )
        lines.append("###########################")
        return "\n".join(lines)

    def print_results(self):
        print(str(self))


class ErrorCategoryCounts(BaseModel):
    ground_truth_counts: dict[str, int]
    prediction_counts: dict[str, int]
    total_turns: int

    def __str__(self) -> str:
        lines = []
        lines.append("######### Error Category Counts #########")
        lines.append(f"Total Turns: {self.total_turns}")
        lines.append("\nGround Truth Counts:")
        for category, count in sorted(self.ground_truth_counts.items()):
            lines.append(f"{category}: {count}")
        lines.append("\nPrediction Counts:")
        for category, count in sorted(self.prediction_counts.items()):
            lines.append(f"{category}: {count}")
        lines.append("###########################")
        return "\n".join(lines)

    def print_results(self):
        print(str(self))

    def plot_counts(self, save_dir: Optional[Path] = None):
        plt.figure(figsize=(10, 6))

        breakdown_taxonomy_titles = get_breakdown_title_list(task_oriented=False)
        categories = [title.lower() for title in breakdown_taxonomy_titles]

        x = range(len(categories))

        width = 0.35
        plt.bar(
            [i - width / 2 for i in x],
            [self.ground_truth_counts.get(cat, 0) for cat in categories],
            width,
            label="Ground Truth",
            color="dodgerblue",
        )
        plt.bar(
            [i + width / 2 for i in x],
            [self.prediction_counts.get(cat, 0) for cat in categories],
            width,
            label="Predicted",
            color="coral",
        )

        plt.xlabel("Error Types")
        plt.ylabel("Count")
        # plt.title("Ground Truth vs Predicted Error Category Counts")
        plt.xticks(x, categories, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        if save_dir:
            # save as pdf and png
            plt.savefig(save_dir / "error_category_counts.pdf")
            plt.savefig(save_dir / "error_category_counts.png")
        plt.show()


def compute_dbdc_error_classification_scores(
    dialogues: List[DBDCErrorClassificationDialogue],
    config_key: str,
    mode: Literal[
        "all_turns", "true_breakdowns", "agreed_breakdowns"
    ] = "agreed_breakdowns",
) -> tuple[ErrorClassificationMetrics, ErrorCategoryCounts, MismatchMetrics]:
    total_turns = 0
    correct_turns = 0
    partial_match_turns = 0
    correct_superset_turns = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_predicted_n_types = 0
    total_ground_truth_n_types = 0

    # Initialize error category counters
    ground_truth_counts: dict[str, int] = {}
    prediction_counts: dict[str, int] = {}
    set_mismatch_counts: dict[frozenset[str], dict[frozenset[str], int]] = {}
    single_mismatch_counts: dict[str, dict[str, int]] = {}

    for dialogue in dialogues:
        for k, turn in enumerate(dialogue.turns):
            if turn.speaker != DBDCSpeakerRole.SYSTEM:
                continue

            # Skip turns without annotations for this config key
            if (
                not turn.llm_breakdown_annotations
                or config_key not in turn.llm_breakdown_annotations
            ):
                continue

            ans_prob_dist = calc_distribution(turn)
            ans_label_l = majority_label_lenient(
                ans_prob_dist[0], ans_prob_dist[1], ans_prob_dist[2], threshold=0.0
            )
            if mode != "all_turns":
                # Skip turns that are not labeled as breakdowns in the ground truth
                if ans_label_l != "X_l":
                    continue

            if mode == "agreed_breakdowns":
                # Skip turns that are not labeled as breakdowns in both the ground truth and the prediction
                if not (
                    ans_label_l == "X_l"
                    and turn.llm_breakdown_annotations[config_key].decision
                    == BreakdownDecision.BREAKDOWN
                ):
                    continue

            # Get reference and predicted error categories
            ref_categories_list = turn.error_category or []
            if mode != "all_turns" and not ref_categories_list:
                # Skip turns that are labeled as breakdowns in the ground truth but have no annotated error categories
                continue

            ref_categories = set(
                [category.strip().lower() for category in ref_categories_list]
            )
            # Rename "ignore offer" to "ignore request" to make it comparable with our naming
            if "ignore offer" in ref_categories:
                ref_categories.remove("ignore offer")
                ref_categories.add("ignore request")

            total_ground_truth_n_types += len(ref_categories)
            pred_categories_list = (
                turn.llm_breakdown_annotations[config_key].breakdown_types or []
            )
            pred_categories = set(
                [
                    category.strip().lower().replace("_", " ")
                    for category in pred_categories_list
                ]
            )

            total_predicted_n_types += len(pred_categories)

            # Calculate exact match
            if ref_categories == pred_categories:
                correct_turns += 1

            # Calculate superset match
            if ref_categories.issubset(pred_categories):
                correct_superset_turns += 1

            # Calculate partial match
            if ref_categories.intersection(pred_categories):
                partial_match_turns += 1

            # Calculate category mismatch counts (mismatches are based on the symmetric difference between the ground truth and the prediction)
            refs_without_pred = frozenset(ref_categories - pred_categories)
            preds_without_ref = frozenset(pred_categories - ref_categories)

            if refs_without_pred and preds_without_ref:
                if refs_without_pred not in set_mismatch_counts:
                    set_mismatch_counts[refs_without_pred] = {}
                set_mismatch_counts[refs_without_pred][preds_without_ref] = (
                    set_mismatch_counts.get(refs_without_pred, {}).get(
                        preds_without_ref, 0
                    )
                    + 1
                )
                for ref_cat in refs_without_pred:
                    if ref_cat not in single_mismatch_counts:
                        single_mismatch_counts[ref_cat] = {}
                    for pred_cat in preds_without_ref:
                        single_mismatch_counts[ref_cat][pred_cat] = (
                            single_mismatch_counts.get(ref_cat, {}).get(pred_cat, 0) + 1
                        )
            # Calculate precision, recall and F1
            if not ref_categories and not pred_categories:
                # Both empty - perfect match
                precision = 1.0
                recall = 1.0
            else:
                # Calculate precision and recall
                if pred_categories:
                    precision = len(ref_categories & pred_categories) / len(
                        pred_categories
                    )
                else:
                    precision = 1.0

                if ref_categories:
                    recall = len(ref_categories & pred_categories) / len(ref_categories)
                else:
                    recall = 1.0

            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            total_turns += 1
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Count ground truth categories
            for category in ref_categories:
                ground_truth_counts[category] = ground_truth_counts.get(category, 0) + 1

            # Count predicted categories
            for category in pred_categories:
                prediction_counts[category] = prediction_counts.get(category, 0) + 1

    # Calculate averages
    avg_precision = total_precision / total_turns if total_turns > 0 else 0.0
    avg_recall = total_recall / total_turns if total_turns > 0 else 0.0
    avg_f1 = total_f1 / total_turns if total_turns > 0 else 0.0
    exact_match = correct_turns / total_turns if total_turns > 0 else 0.0
    superset_match = correct_superset_turns / total_turns if total_turns > 0 else 0.0
    partial_match = partial_match_turns / total_turns if total_turns > 0 else 0.0
    predicted_n_types_per_turn = (
        total_predicted_n_types / total_turns if total_turns > 0 else 0.0
    )
    ground_truth_n_types_per_turn = (
        total_ground_truth_n_types / total_turns if total_turns > 0 else 0.0
    )

    return (
        ErrorClassificationMetrics(
            exact_match=exact_match,
            superset_match=superset_match,
            partial_match=partial_match,
            f1_score=avg_f1,
            precision=avg_precision,
            recall=avg_recall,
            total_turns=total_turns,
            correct_turns=correct_turns,
            correct_superset_turns=correct_superset_turns,
            partial_match_turns=partial_match_turns,
            predicted_n_types_per_turn=predicted_n_types_per_turn,
            ground_truth_n_types_per_turn=ground_truth_n_types_per_turn,
        ),
        ErrorCategoryCounts(
            ground_truth_counts=ground_truth_counts,
            prediction_counts=prediction_counts,
            total_turns=total_turns,
        ),
        MismatchMetrics(
            set_mismatch_counts=set_mismatch_counts,
            single_mismatch_counts=single_mismatch_counts,
        ),
    )
