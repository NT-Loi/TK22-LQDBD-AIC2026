from .prompt import parse_aspects, build_shot_prompt

__all__ = ["parse_aspects", "build_shot_prompt", "ShotCaptionGenerator"]


def __getattr__(name: str):
    if name == "ShotCaptionGenerator":
        from .caption_generator import ShotCaptionGenerator
        return ShotCaptionGenerator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
