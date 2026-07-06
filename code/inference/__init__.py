# inference package — Delphos v1.2
from .inference_zero_shot_learning import run_zero_shot_inference
from .inference_few_shot_learning import run_few_shot_inference

__all__ = [
    "run_zero_shot_inference",
    "run_few_shot_inference",
]
