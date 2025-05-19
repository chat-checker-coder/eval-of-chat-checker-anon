from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import random

from chat_checker.models.chatbot import ChatbotInfo
from chat_checker.models.rating import RatingDimension
from models.benchmark_dialogues import EvaluatedDialogue


@dataclass
class RatingBenchmarkDataset:
    name: str
    all_samples: list[EvaluatedDialogue]
    rated_samples_path: Path
    chatbot_info: ChatbotInfo
    rating_dimensions: list[RatingDimension]
    label_aggregation_method: str = "mean"

    def load_rated_samples(self) -> list[EvaluatedDialogue]:
        if self.rated_samples_path.exists():
            with open(self.rated_samples_path, "r", encoding="utf-8") as f:
                rated_samples = json.load(f)
        else:
            rated_samples = []
        return [EvaluatedDialogue(**d) for d in rated_samples]

    @cached_property
    def representative_few_shot_samples(self) -> list[EvaluatedDialogue]:
        sorted_dataset = sorted(
            self.all_samples,
            key=lambda x: x.rating_annotations["overall"].avg_rating or 3,
        )

        low_rating_sample = sorted_dataset[0]
        high_rating_sample = sorted_dataset[-1]
        median_rating_sample = sorted_dataset[len(sorted_dataset) // 2]

        return [
            low_rating_sample,
            median_rating_sample,
            high_rating_sample,
        ]

    @cached_property
    def random_few_shot_samples(self) -> list[EvaluatedDialogue]:
        return random.sample(self.all_samples, 3)
