"""Model registry for controlled experiment selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for one supported model choice."""

    choice: str
    model_id: str
    family: str
    size_label: str
    notes: str

    @property
    def slug(self) -> str:
        """Return a filesystem-friendly model slug."""
        return self.model_id.split("/")[-1].replace(".", "-").lower()


GEMMA2_MODEL_SPECS: dict[str, ModelSpec] = {
    "Gemma 2 2B IT": ModelSpec(
        choice="Gemma 2 2B IT",
        model_id="google/gemma-2-2b-it",
        family="gemma2",
        size_label="2B",
        notes="Stage 1 anchor baseline; expected to run full PKV profiling on T4 FP16.",
    ),
    "Gemma 2 9B IT": ModelSpec(
        choice="Gemma 2 9B IT",
        model_id="google/gemma-2-9b-it",
        family="gemma2",
        size_label="9B",
        notes="Medium dense model for T4 capacity and memory pressure probing.",
    ),
    "Gemma 2 27B IT": ModelSpec(
        choice="Gemma 2 27B IT",
        model_id="google/gemma-2-27b-it",
        family="gemma2",
        size_label="27B",
        notes="Large dense model; expected to expose T4 FP16 load or inference limits.",
    ),
}


def get_model_spec(choice: str) -> ModelSpec:
    """Return the model spec for a UI choice."""
    try:
        return GEMMA2_MODEL_SPECS[choice]
    except KeyError as exc:
        valid = ", ".join(GEMMA2_MODEL_SPECS)
        raise ValueError(f"Unsupported model choice: {choice}. Valid choices: {valid}") from exc


def model_choices() -> list[str]:
    """Return stable UI choices for Colab forms."""
    return list(GEMMA2_MODEL_SPECS)
