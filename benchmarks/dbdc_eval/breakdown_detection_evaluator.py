#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Adapted from https://github.com/dbd-challenge/dbdc4/tree/aecf252505e9b1736723848b78eb55aaf5712b4d/prog/eval

import math
from typing import List

from pydantic import BaseModel
from models.benchmark_dialogues import (
    DBDCDialogue,
    DBDCBreakdownCategory,
    DBDCBreakdownAnnotation,
    DBDCSpeakerRole,
)


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

    # Per-dialogue metrics (included to compare with Ghassel et al. (https://github.com/aghassel/LLM-dialogue-breakdown-detection-challenge/blob/main/Evaluation/evaluation.ipynb))
    avg_f1_per_dialogue: float
    avg_lenient_accuracy_per_dialogue: float

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
        lines.append("Avg F1 per dialogue : \t%4f" % self.avg_f1_per_dialogue + "\n")
        lines.append(
            "Avg Lenient Accuracy per dialogue : \t%4f"
            % self.avg_lenient_accuracy_per_dialogue
            + "\n"
        )

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


def calc_distribution(annotations: List[DBDCBreakdownAnnotation]) -> List[float]:
    count_O = 0
    count_T = 0
    count_X = 0

    for annotation in annotations:
        if annotation.breakdown == DBDCBreakdownCategory.NO_BREAKDOWN:
            count_O += 1
        elif annotation.breakdown == DBDCBreakdownCategory.POSSIBLE_BREAKDOWN:
            count_T += 1
        elif annotation.breakdown == DBDCBreakdownCategory.BREAKDOWN:
            count_X += 1

    total = count_O + count_T + count_X
    if total == 0:
        return [0.0, 0.0, 0.0]

    prob_O = count_O * 1.0 / total
    prob_T = count_T * 1.0 / total
    prob_X = count_X * 1.0 / total

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


def compute_dbdc_scores(
    dialogues: List[DBDCDialogue], config_key: str, threshold: float = 0.0
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

    # Per-dialogue metrics
    dialogue_f1_scores = []
    dialogue_accuracy_scores = []

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

        # Per-dialogue counters
        dialogue_true_positives = 0
        dialogue_false_positives = 0
        dialogue_false_negatives = 0
        dialogue_correct_count = 0
        dialogue_total_count = 0

        for turn in dialogue.turns:
            if turn.speaker != DBDCSpeakerRole.SYSTEM or not turn.annotations:
                continue

            label_count += 1

            ans_prob_dist = calc_distribution(turn.annotations)
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
                    dialogue_correct_count += 1
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
                    dialogue_correct_count += 1
            elif pred_label == DBDCBreakdownCategory.BREAKDOWN:
                if ans_label_l == "O_l":
                    predX_ansO_l += 1
                    lenient_incorrect_count += 1
                elif ans_label_l == "X_l":
                    predX_ansX_l += 1
                    lenient_correct_count += 1
                    dialogue_correct_count += 1

            dialogue_total_count += 1

            # Update per-dialogue lenient metrics
            if pred_label == DBDCBreakdownCategory.NO_BREAKDOWN:
                if ans_label_l == "X_l":
                    dialogue_false_negatives += 1
            elif pred_label in [
                DBDCBreakdownCategory.POSSIBLE_BREAKDOWN,
                DBDCBreakdownCategory.BREAKDOWN,
            ]:
                if ans_label_l == "O_l":
                    dialogue_false_positives += 1
                elif ans_label_l == "X_l":
                    dialogue_true_positives += 1

        # Calculate F1 score for this dialogue
        dialogue_precision = (
            dialogue_true_positives
            / (dialogue_true_positives + dialogue_false_positives)
            if (dialogue_true_positives + dialogue_false_positives) > 0
            else 0.0
        )
        dialogue_recall = (
            dialogue_true_positives
            / (dialogue_true_positives + dialogue_false_negatives)
            if (dialogue_true_positives + dialogue_false_negatives) > 0
            else 0.0
        )
        dialogue_f1 = (
            2
            * dialogue_precision
            * dialogue_recall
            / (dialogue_precision + dialogue_recall)
            if (dialogue_precision + dialogue_recall) > 0
            else 0.0
        )
        dialogue_f1_scores.append(dialogue_f1)

        # Calculate accuracy for this dialogue
        dialogue_accuracy = (
            dialogue_correct_count / dialogue_total_count
            if dialogue_total_count > 0
            else 0.0
        )
        dialogue_accuracy_scores.append(dialogue_accuracy)

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

    # Calculate average F1 score per dialogue
    avg_f1_per_dialogue = (
        sum(dialogue_f1_scores) / len(dialogue_f1_scores) if dialogue_f1_scores else 0.0
    )

    # Calculate average lenient accuracy per dialogue
    avg_lenient_accuracy_per_dialogue = (
        sum(dialogue_accuracy_scores) / len(dialogue_accuracy_scores)
        if dialogue_accuracy_scores
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
        avg_f1_per_dialogue=avg_f1_per_dialogue,
        avg_lenient_accuracy_per_dialogue=avg_lenient_accuracy_per_dialogue,
    )
