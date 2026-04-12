"""YAML loader for fixed-stage curricula."""

from pathlib import Path

import yaml

from curriculum.models import CurriculumStage, OpponentPlayerSpec
from curriculum.runtime import LinearCurriculum


def load_curriculum_from_yaml(path: str | Path) -> LinearCurriculum:
    """Load a linear curriculum from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    stages_raw = _extract_stages(raw_config)
    stages = _parse_stages(stages_raw)
    return LinearCurriculum(stages)


def _extract_stages(raw_config) -> list[dict]:
    if isinstance(raw_config, list):
        stages = raw_config
    elif isinstance(raw_config, dict):
        stages = raw_config.get("stages")
    else:
        raise ValueError("Curriculum YAML must be a list or a mapping with a 'stages' key.")

    if not isinstance(stages, list) or not stages:
        raise ValueError("Curriculum YAML must define at least one stage.")
    return stages


def _parse_stages(stages_raw: list[dict]) -> list[CurriculumStage]:
    stages: list[CurriculumStage] = []
    start_timestep = 0
    last_index = len(stages_raw) - 1

    for index, stage_raw in enumerate(stages_raw):
        if not isinstance(stage_raw, dict):
            raise ValueError("Each curriculum stage must be a mapping.")

        name = _require_string(stage_raw, "name")
        opponent_raw = stage_raw.get("opponent_player")
        if not isinstance(opponent_raw, dict):
            raise ValueError(
                f"Stage '{name}' must define an 'opponent_player' mapping."
            )

        has_duration = "duration_timesteps" in stage_raw
        has_end = "end_timestep" in stage_raw
        if has_duration and has_end:
            raise ValueError(
                f"Stage '{name}' cannot define both duration_timesteps and end_timestep."
            )
        if not has_duration and not has_end and index != last_index:
            raise ValueError(
                f"Stage '{name}' must define duration_timesteps or end_timestep unless it is the final stage."
            )

        end_timestep: int | None
        if has_duration:
            duration = _require_positive_int(stage_raw, "duration_timesteps")
            end_timestep = start_timestep + duration
        elif has_end:
            end_timestep = _require_positive_int(stage_raw, "end_timestep")
            if end_timestep <= start_timestep:
                raise ValueError(
                    f"Stage '{name}' end_timestep must be greater than the previous stage boundary."
                )
        else:
            end_timestep = None

        stages.append(
            CurriculumStage(
                name=name,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                opponent_player=_parse_opponent_player(opponent_raw),
            )
        )

        if end_timestep is not None:
            start_timestep = end_timestep

    return stages


def _parse_opponent_player(opponent_raw: dict) -> OpponentPlayerSpec:
    player_id = opponent_raw.get("id")
    class_path = opponent_raw.get("class_path")
    kwargs = opponent_raw.get("kwargs", {})

    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("opponent_player.kwargs must be a mapping if provided.")

    return OpponentPlayerSpec(id=player_id, class_path=class_path, kwargs=kwargs)


def _require_string(stage_raw: dict, field_name: str) -> str:
    value = stage_raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Curriculum stage field '{field_name}' must be a non-empty string.")
    return value


def _require_positive_int(stage_raw: dict, field_name: str) -> int:
    value = stage_raw.get(field_name)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Curriculum stage field '{field_name}' must be a positive integer.")
    return value
