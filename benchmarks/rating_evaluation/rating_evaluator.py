import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from litellm import completion_cost
import matplotlib.pyplot as plt

from chat_checker.dialogue_rating.dialogue_rater import get_dialogue_rating
from chat_checker.utils.misc_utils import write_prompt_to_txt_file
from models.benchmark_dialogues import EvaluatedDialogue
from models.configs import RatingEvalConfig
from models.datasets import RatingBenchmarkDataset

plt.style.use("seaborn-v0_8")


def compute_llm_ratings(
    dialogues: list[EvaluatedDialogue],
    config: RatingEvalConfig,
    recompute_existing: bool = False,
    debug_dir: Optional[Path] = None,
):
    for i, dialogue in tqdm(enumerate(dialogues)):
        has_llm_rating = (
            dialogue.llm_rating_annotations is not None
            and dialogue.llm_rating_annotations.get(config.key) is not None
        )
        if has_llm_rating and not recompute_existing:
            continue

        ratings, prompt, model_response = get_dialogue_rating(
            dialogue.chat_history,
            rating_dimensions=config.rating_dimensions,
            chatbot_info=config.chatbot_info,
            examples=config.few_shot_samples or [],
            rating_model=config.model,
        )

        if not dialogue.llm_rating_annotations:
            dialogue.llm_rating_annotations = {}
        dialogue.llm_rating_annotations[config.key] = ratings

        if i == 0 and debug_dir:
            # Save the prompt and model response for the first dialogue
            debug_dir.mkdir(parents=True, exist_ok=True)
            write_prompt_to_txt_file(prompt, debug_dir / "sample_0_prompt.txt")

            with open(
                debug_dir / "sample_0_model_response.json", "w", encoding="utf-8"
            ) as f:
                json.dump(model_response.model_dump(), f, ensure_ascii=False, indent=2)
            # Save the cost of the model response
            cost = completion_cost(model_response)
            with open(
                debug_dir / "sample_0_response_cost.txt", "w", encoding="utf-8"
            ) as f:
                f.write(f"Model response cost: {cost:.8f} USD\n")


def expand_rating_and_compute_ensemble_rating(sample: dict):
    rating = sample["llm_rating_annotations"]
    dimension_ratings = []
    for key, value in rating.items():
        element_rating = value.rating
        sample[f"llm_{key}"] = element_rating
        dimension_ratings.append(element_rating)
    if len(dimension_ratings) == 0:
        dimension_ensemble_rating = None
    else:
        dimension_ensemble_rating = sum(dimension_ratings) / len(dimension_ratings)
    sample["llm_dimension_ensemble_rating"] = dimension_ensemble_rating
    return sample


def compute_correlations(
    dialogues: list[EvaluatedDialogue],
    config: RatingEvalConfig,
    label_aggregation_method: str = "mean",
) -> dict[str, Optional[float]]:
    def build_df_dict(dialogue: EvaluatedDialogue):
        if dialogue.llm_rating_annotations is None:
            raise ValueError(
                f"Dialogue {dialogue.dialogue_id} does not have LLM ratings"
            )

        return {
            "dialogue_id": dialogue.dialogue_id,
            "avg_rating": dialogue.rating_annotations["overall"].avg_rating,
            "mode_rating": dialogue.rating_annotations["overall"].mode_rating,
            "llm_overall_performance": dialogue.llm_rating_annotations[config.key][
                "overall_performance"
            ].rating,
            "llm_rating_annotations": dialogue.llm_rating_annotations[config.key],
        }

    subset_df: pd.DataFrame = pd.DataFrame(
        [build_df_dict(sample) for sample in dialogues]
    )
    subset_df = subset_df.apply(expand_rating_and_compute_ensemble_rating, axis=1)

    if label_aggregation_method == "mean":
        gt_rating_key = "avg_rating"
    elif label_aggregation_method == "mode":
        gt_rating_key = "mode_rating"
    else:
        raise ValueError(
            f"Invalid label aggregation method: {label_aggregation_method}"
        )

    pearson_correlation_overall = subset_df[gt_rating_key].corr(
        subset_df["llm_overall_performance"], method="pearson"
    )
    kendall_correlation_overall = subset_df[gt_rating_key].corr(
        subset_df["llm_overall_performance"], method="kendall"
    )
    spearman_correlation_overall = subset_df[gt_rating_key].corr(
        subset_df["llm_overall_performance"], method="spearman"
    )

    pearson_correlation_ensemble = subset_df[gt_rating_key].corr(
        subset_df["llm_dimension_ensemble_rating"], method="pearson"
    )
    kendall_correlation_ensemble = subset_df[gt_rating_key].corr(
        subset_df["llm_dimension_ensemble_rating"], method="kendall"
    )
    spearman_correlation_ensemble = subset_df[gt_rating_key].corr(
        subset_df["llm_dimension_ensemble_rating"], method="spearman"
    )

    return {
        "pearson_correlation_overall": pearson_correlation_overall,
        "kendall_correlation_overall": kendall_correlation_overall,
        "spearman_correlation_overall": spearman_correlation_overall,
        "person_correlation_dimension_ensemble": pearson_correlation_ensemble
        if not np.isnan(pearson_correlation_ensemble)
        else None,
        "kendall_correlation_dimension_ensemble": kendall_correlation_ensemble
        if not np.isnan(kendall_correlation_ensemble)
        else None,
        "spearman_correlation_dimension_ensemble": spearman_correlation_ensemble
        if not np.isnan(spearman_correlation_ensemble)
        else None,
    }


def plot_human_vs_llm_scatter(
    dialogues: list[EvaluatedDialogue],
    config: RatingEvalConfig,
    save_path: Path,
    label_aggregation_method: str = "mean",
):
    """
    Creates and saves a scatter plot comparing the human average rating
    and the LLM's overall performance rating for a set of dialogues.
    Points get larger when multiple data points overlap.
    """
    print("Plotting ratings in scatter plot...")

    human_ratings = []
    if label_aggregation_method == "mean":
        human_ratings = [
            dialogue.rating_annotations["overall"].avg_rating
            for dialogue in dialogues
            if dialogue.llm_rating_annotations is not None
        ]
    elif label_aggregation_method == "mode":
        human_ratings = [
            dialogue.rating_annotations["overall"].mode_rating
            for dialogue in dialogues
            if dialogue.llm_rating_annotations is not None
        ]
    else:
        raise ValueError(
            f"Invalid label aggregation method: {label_aggregation_method}"
        )

    # Build a DataFrame with human and LLM ratings
    df = pd.DataFrame(
        {
            "human_rating": human_ratings,
            "llm_rating": [
                dialogue.llm_rating_annotations[config.key][
                    "overall_performance"
                ].rating
                for dialogue in dialogues
                if dialogue.llm_rating_annotations is not None
            ],
        }
    )

    plt.figure()
    # plt.xlim(0, 6)
    # plt.ylim(0, 6)

    # Count frequency of each x,y coordinate pair
    counts = df.groupby(["human_rating", "llm_rating"]).size().reset_index(name="count")

    # Create scatter plot with size proportional to count
    plt.scatter(
        counts["human_rating"],
        counts["llm_rating"],
        s=counts["count"] * 100,  # Multiply by 100 to make differences more visible
        alpha=0.6,
        color="dodgerblue",
    )

    plt.xlabel("Human Average Rating", fontsize="large")
    plt.ylabel("LLM Overall Performance Rating", fontsize="large")
    # plt.title("Scatter Plot of Human vs LLM Overall Ratings")
    plt.grid(True)
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved to {save_path}")


def plot_llm_ratings_barplots(
    dialogues: list[EvaluatedDialogue],
    config: RatingEvalConfig,
    save_path: Path,
):
    """
    Creates and saves bar plots for the rating distribution of each LLM rating dimension.
    Each bar plot shows the frequency of integer ratings (1 through 5).
    """
    # Identify rating dimensions from the first dialogue's annotations
    if not dialogues or not dialogues[0].llm_rating_annotations:
        raise ValueError("No dialogues or no LLM rating annotations found.")
    dimensions = list(dialogues[0].llm_rating_annotations[config.key].keys())
    ratings_dict: dict[str, list[int]] = {dim: [] for dim in dimensions}

    # Collect ratings for each dimension, ignoring None values
    for dialogue in dialogues:
        if not dialogue.llm_rating_annotations:
            print(
                f"Skipping dialogue {dialogue.dialogue_id} due to missing LLM ratings."
            )
            continue
        annotations = dialogue.llm_rating_annotations[config.key]
        for dim in dimensions:
            rating = annotations[dim].rating
            if rating is not None:
                ratings_dict[dim].append(rating)

    # Determine grid size: two columns if possible
    num_dims = len(dimensions)
    ncols = 2
    nrows = (num_dims + 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axs = axs.flatten()
    # # Flatten the axes array for easy iteration
    # if num_dims > 1:
    #     axs = axs.flatten()
    # else:
    #     axs = [axs]

    for idx, dim in enumerate(dimensions):
        # Compute frequency counts for ratings 1 to 5
        counts = [ratings_dict[dim].count(i) for i in range(1, 6)]
        axs[idx].bar(range(1, 6), counts, edgecolor="black")
        axs[idx].set_xticks(range(1, 6))
        axs[idx].set_xlabel("Rating")
        axs[idx].set_ylabel("Frequency")
        axs[idx].set_title(f"Bar Plot for {dim}")

    # Remove any unused subplots if they exist
    for idx in range(num_dims, len(axs)):
        fig.delaxes(axs[idx])

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Bar plots saved to {save_path}")


def evaluate_ratings(
    dialogues: list[EvaluatedDialogue],
    config: RatingEvalConfig,
    benchmark_dataset: RatingBenchmarkDataset,
    recompute_existing_ratings: bool = False,
) -> dict[str, Optional[float]]:
    rated_samples_file = benchmark_dataset.rated_samples_path
    config_res_path = rated_samples_file.parent / config.key
    compute_llm_ratings(
        dialogues,
        config,
        recompute_existing=recompute_existing_ratings,
        debug_dir=config_res_path,
    )

    # Save the rated subset to a file
    print(f"Saving rated subset to {rated_samples_file}...")
    rated_samples_file.parent.mkdir(parents=True, exist_ok=True)
    # Load previous rated samples if they exist
    if rated_samples_file.exists():
        with open(rated_samples_file, "r", encoding="utf-8") as f:
            previous_rated_samples = json.load(f)
        previous_rated_samples = [
            EvaluatedDialogue(**sample) for sample in previous_rated_samples
        ]
        # Filter out the previously rated samples that are not in the current set
        dialogue_ids_map = {sample.dialogue_id: sample for sample in dialogues}
        previous_rated_samples = [
            sample
            for sample in previous_rated_samples
            if sample.dialogue_id not in dialogue_ids_map
        ]
        # Merge the previous rated samples with the current set
        dialogues_to_save = dialogues + previous_rated_samples
    else:
        dialogues_to_save = dialogues
    # Save the rated samples to a file
    with open(rated_samples_file, "w", encoding="utf-8") as f:
        json.dump(
            [dialogue.model_dump() for dialogue in dialogues_to_save],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved {len(dialogues)} rated samples to {rated_samples_file}")

    # Compute correlations
    print("Computing correlations...")
    correlations = compute_correlations(
        dialogues,
        config,
        label_aggregation_method=benchmark_dataset.label_aggregation_method,
    )
    print("Correlations computed")
    print(json.dumps(correlations, indent=2))

    # Save the correlations to a file
    correlations_path = config_res_path / "correlations.json"
    print(f"Saving correlations to {correlations_path}...")
    correlations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(correlations_path, "w", encoding="utf-8") as f:
        json.dump(correlations, f, ensure_ascii=False, indent=2)
    print(f"Saved correlations to {correlations_path}")

    # Create and save the scatter plot of human rating vs LLM overall rating
    scatter_plot_path = config_res_path / "scatter_plot.png"
    plot_human_vs_llm_scatter(
        dialogues,
        config,
        save_path=scatter_plot_path,
        label_aggregation_method=benchmark_dataset.label_aggregation_method,
    )

    # Create and save the bar plots for LLM rating distributions for each dimension
    bar_plots_path = config_res_path / "llm_ratings_barplots.png"
    plot_llm_ratings_barplots(dialogues, config, bar_plots_path)

    return correlations
