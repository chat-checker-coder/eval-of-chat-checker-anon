from dataclasses import dataclass
from typing import Optional

from chat_checker.breakdown_detection.breakdown_detector import BreakdownIdentifier
from chat_checker.models.chatbot import ChatbotInfo
from chat_checker.models.dialogue import Dialogue
from chat_checker.models.rating import RatingDimension


@dataclass
class RatingEvalConfig:
    key: str
    model: str
    rating_dimensions: list[RatingDimension]
    few_shot_samples: Optional[list[Dialogue]]
    chatbot_info: Optional[ChatbotInfo]


@dataclass
class BreakdownDetectionConfig:
    key: str
    model: str
    breakdown_identifier: BreakdownIdentifier
    include_task_oriented_errors: bool
