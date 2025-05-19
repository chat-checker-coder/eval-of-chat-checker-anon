from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from chat_checker.models.breakdowns import BreakdownAnnotation, BreakdownDecision
from chat_checker.models.dialogue import (
    Dialogue,
    DialogueTurn,
    FinishReason,
    SpeakerRole,
)
from chat_checker.models.rating import (
    DialogueDimensionRating,
    RatingDimensionAnnotation,
)


class EvaluatedDialogue(BaseModel):
    dialogue_id: str
    chat_history: list[DialogueTurn]
    rating_annotations: dict[str, RatingDimensionAnnotation]
    chat_model: Optional[str] = None
    llm_rating_annotations: Optional[dict[str, dict[str, DialogueDimensionRating]]] = (
        None
    )

    def to_chat_checker_dialogue(self) -> Dialogue:
        return Dialogue(
            dialogue_id=self.dialogue_id,
            path=Path(),
            user_name="",
            finish_reason=FinishReason.USER_ENDED,
            chat_history=self.chat_history,
            human_rating_annotations=self.rating_annotations,
        )


class DBDCSpeakerRole(StrEnum):
    USER = "U"
    SYSTEM = "S"


class DBDCBreakdownCategory(StrEnum):
    BREAKDOWN = "X"
    POSSIBLE_BREAKDOWN = "T"
    NO_BREAKDOWN = "O"


class DBDCEvalPrediction(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    breakdown: DBDCBreakdownCategory
    prob_O: float
    prob_T: float
    prob_X: float


class DBDCPredictionTurn(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    turn_index: int
    labels: list[DBDCEvalPrediction]


class DBDCPredictionsDialogue(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    dialogue_id: str
    speaker_id: str
    turns: list[DBDCPredictionTurn]


class DBDCBreakdownAnnotation(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    comment: str
    breakdown: DBDCBreakdownCategory
    annotator_id: str
    ungrammatical_sentence: str


class DBDCDialogueTurn(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    turn_index: int
    speaker: DBDCSpeakerRole
    time: Optional[str] = None
    annotation_id: Optional[str] = None
    utterance: str
    annotations: list[DBDCBreakdownAnnotation]
    llm_breakdown_annotations: Optional[dict[str, BreakdownAnnotation]] = None

    def to_chat_checker_turn(self) -> DialogueTurn:
        return DialogueTurn(
            turn_id=self.turn_index,
            role=SpeakerRole.USER
            if self.speaker == DBDCSpeakerRole.USER
            else SpeakerRole.DIALOGUE_SYSTEM,
            content=self.utterance,
            breakdown_annotation=None,
        )

    def to_eval_prediction_turn(self, key: str) -> DBDCPredictionTurn:
        if (
            not self.llm_breakdown_annotations
            or key not in self.llm_breakdown_annotations
        ):
            raise ValueError(f"Key {key} not found in llm_breakdown_annotations")
        breakdown_annotation = self.llm_breakdown_annotations[key]
        # Note: our llm detector only distinguishes between breakdown and no breakdown
        breakdown = (
            DBDCBreakdownCategory.BREAKDOWN
            if breakdown_annotation.decision == BreakdownDecision.BREAKDOWN
            else DBDCBreakdownCategory.NO_BREAKDOWN
        )

        # Use the correct field names with the actual aliases that Pydantic expects
        label = DBDCEvalPrediction(
            breakdown=breakdown,
            **{
                "prob-O": breakdown_annotation.score,
                "prob-T": float(0),
                "prob-X": 1 - breakdown_annotation.score,
            },
        )
        return DBDCPredictionTurn(labels=[label], **{"turn-index": self.turn_index})


class DBDCDialogue(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    dialogue_id: str
    speaker_id: str
    group_id: str
    turns: list[DBDCDialogueTurn]

    def to_chat_checker_dialogue(self) -> Dialogue:
        return Dialogue(
            dialogue_id=self.dialogue_id,
            path=Path(),
            user_name="",
            finish_reason=FinishReason.USER_ENDED,
            chat_history=[turn.to_chat_checker_turn() for turn in self.turns],
        )

    def to_eval_prediction_dialogue(self, key: str) -> DBDCPredictionsDialogue:
        return DBDCPredictionsDialogue(
            turns=[
                turn.to_eval_prediction_turn(key)
                for turn in self.turns
                if turn.speaker == DBDCSpeakerRole.SYSTEM
            ],
            **{"dialogue-id": self.dialogue_id, "speaker-id": self.speaker_id},
        )


class DBDCErrorClassificationTurn(BaseModel):
    num_annotations: int = Field(alias="#annotation")
    num_O: int = Field(alias="#O")
    num_T: int = Field(alias="#T")
    num_X: int = Field(alias="#X")
    error_category: Optional[list[str]] = None
    speaker: str
    time: str
    turn_index: int = Field(alias="turn-index")
    utterance: str
    llm_breakdown_annotations: Optional[dict[str, BreakdownAnnotation]] = None

    def to_chat_checker_turn(self) -> DialogueTurn:
        return DialogueTurn(
            turn_id=self.turn_index,
            role=SpeakerRole.USER
            if self.speaker == DBDCSpeakerRole.USER
            else SpeakerRole.DIALOGUE_SYSTEM,
            content=self.utterance,
            breakdown_annotation=None,
        )

    def to_eval_prediction_turn(self, key: str) -> DBDCPredictionTurn:
        if (
            not self.llm_breakdown_annotations
            or key not in self.llm_breakdown_annotations
        ):
            raise ValueError(f"Key {key} not found in llm_breakdown_annotations")
        breakdown_annotation = self.llm_breakdown_annotations[key]
        # Note: our llm detector only distinguishes between breakdown and no breakdown
        breakdown = (
            DBDCBreakdownCategory.BREAKDOWN
            if breakdown_annotation.decision == BreakdownDecision.BREAKDOWN
            else DBDCBreakdownCategory.NO_BREAKDOWN
        )

        # Use the correct field names with the actual aliases that Pydantic expects
        label = DBDCEvalPrediction(
            breakdown=breakdown,
            **{
                "prob-O": breakdown_annotation.score,
                "prob-T": float(0),
                "prob-X": 1 - breakdown_annotation.score,
            },
        )
        return DBDCPredictionTurn(labels=[label], **{"turn-index": self.turn_index})


class DBDCErrorClassificationDialogue(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.replace("_", "-"),
    )

    dialogue_id: str
    speaker_id: str
    group_id: str
    turns: list[DBDCErrorClassificationTurn]

    def to_chat_checker_dialogue(self) -> Dialogue:
        return Dialogue(
            dialogue_id=self.dialogue_id,
            path=Path(),
            user_name="",
            finish_reason=FinishReason.USER_ENDED,
            chat_history=[turn.to_chat_checker_turn() for turn in self.turns],
        )

    def to_eval_prediction_dialogue(self, key: str) -> DBDCPredictionsDialogue:
        return DBDCPredictionsDialogue(
            turns=[
                turn.to_eval_prediction_turn(key)
                for turn in self.turns
                if turn.speaker == DBDCSpeakerRole.SYSTEM
            ],
            **{"dialogue-id": self.dialogue_id, "speaker-id": self.speaker_id},
        )
