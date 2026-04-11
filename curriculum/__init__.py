"""Curriculum support for staged RL training."""

from curriculum.models import CurriculumStage, OpponentPlayerSpec
from curriculum.runtime import Curriculum, LinearCurriculum
from curriculum.yaml_loader import load_curriculum_from_yaml

__all__ = [
    "Curriculum",
    "CurriculumStage",
    "LinearCurriculum",
    "OpponentPlayerSpec",
    "load_curriculum_from_yaml",
]
