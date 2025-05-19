from dataclasses import dataclass
from matplotlib import pyplot as plt
from models.benchmark_dialogues import EvaluatedDialogue
# import scienceplots

# plt.style.use(["science", "no-latex"])
plt.style.use("seaborn-v0_8")


@dataclass
class DatasetStatistics:
    num_dialogues: int
    avg_turns_per_dialogue: float
    avg_system_words: float
    avg_user_words: float
    avg_rating: float
    rating_distribution: dict[int, int]

    def __str__(self) -> str:
        lines = []
        lines.append("Dataset Statistics:")
        lines.append(f"Number of dialogues: {self.num_dialogues}")
        lines.append(f"Average turns per dialogue: {self.avg_turns_per_dialogue:.2f}")
        lines.append(f"Average words per user utterance: {self.avg_user_words:.2f}")
        lines.append(f"Average words per system utterance: {self.avg_system_words:.2f}")
        lines.append(f"Average rating per dialogue: {self.avg_rating:.2f}")
        lines.append("\nRating Distribution:")
        for rating, count in sorted(self.rating_distribution.items()):
            lines.append(
                f"  Rating {rating}: {count} dialogues ({(count/self.num_dialogues)*100:.2f}%)"
            )
        return "\n".join(lines)


def compute_dataset_statistics(samples: list[EvaluatedDialogue]) -> DatasetStatistics:
    """
    Compute statistics for the dataset:
    - number of dialogues
    - avg. number of turns per dialogue
    - avg. words per system utterance
    - avg. words per user utterance
    - avg. rating per dialogue
    - distribution of mode human ratings for the overall rating
    """
    num_dialogues = len(samples)
    total_turns = 0
    total_system_words = 0
    total_user_words = 0
    total_system_turns = 0
    total_user_turns = 0
    total_ratings = 0
    total_rated_dialogues = 0
    rating_counts: dict[int, int] = {}

    for sample in samples:
        total_turns += len(sample.chat_history)

        # Count words per utterance type
        for turn in sample.chat_history:
            words = len(turn.content.split())
            if turn.role == "dialogue_system":
                total_system_words += words
                total_system_turns += 1
            else:  # user
                total_user_words += words
                total_user_turns += 1

        # Get overall rating
        if "overall" in sample.rating_annotations:
            rating = sample.rating_annotations["overall"].mode_rating
            if rating is not None:
                total_rated_dialogues += 1
                total_ratings += rating
                rating_counts[rating] = rating_counts.get(rating, 0) + 1

    # Calculate averages
    avg_turns_per_dialogue = total_turns / num_dialogues if num_dialogues > 0 else 0
    avg_system_words = (
        total_system_words / total_system_turns if total_system_turns > 0 else 0
    )
    avg_user_words = total_user_words / total_user_turns if total_user_turns > 0 else 0
    avg_rating = (
        total_ratings / total_rated_dialogues if total_rated_dialogues > 0 else 0
    )

    return DatasetStatistics(
        num_dialogues=num_dialogues,
        avg_turns_per_dialogue=avg_turns_per_dialogue,
        avg_system_words=avg_system_words,
        avg_user_words=avg_user_words,
        avg_rating=avg_rating,
        rating_distribution=rating_counts,
    )


def plot_human_rating_bar_plots(samples: list[EvaluatedDialogue]):
    """
    Plot bar plot for mode human ratings for the overall rating
    """

    # Get the mode human ratings for the overall rating
    overall_ratings = [
        sample.rating_annotations["overall"].mode_rating for sample in samples
    ]
    filtered_overall_ratings = [
        rating for rating in overall_ratings if rating is not None
    ]

    # Count occurrences of each rating
    unique_ratings = sorted(set(filtered_overall_ratings))
    rating_counts = [overall_ratings.count(rating) for rating in unique_ratings]

    # Plot the bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(unique_ratings, rating_counts)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()
