"""
experiments/training.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: High-level training experiment runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from mdp.catalogue import Catalogue
from agent.delphos import Delphos
from agent.replay_buffer import ReplayBuffer
from mdp.runtime import build_runtime
# pyrefly: ignore [missing-import]
from training.trainer import Trainer

from diagnostics.diagnostics_logger import DiagnosticsLogger
from diagnostics.diagnostics_summary import export_all_csvs, summarize_training
# pyrefly: ignore [missing-import]

from dataclasses import dataclass, field

@dataclass
class TrainingExperimentResult:
    trainer: object
    checkpoint_path: str | None = None
    diagnostics_summary: dict = field(default_factory=dict)

def run_training_experiment(
    tasks: list,
    catalogue: Catalogue,
    agent: Delphos,
    replay_buffer: ReplayBuffer,
    epsilon_schedule: dict[float, int],
    batch_size: int = 64,
    horizon_kappa: float = 2.0,
    checkpoint_dir: str = "./checkpoints",
    diagnostics: bool = True,
    seed: Optional[int] = 123,
) -> TrainingExperimentResult:


    runtimes = {int(task.id): build_runtime(task=task, catalogue=catalogue, linear_additive=False, device=str(agent.device)) for task in tasks}

    trainer = Trainer(
        agent=agent, 
        runtimes=runtimes, 
        replay_buffer=replay_buffer, 
        epsilon_schedule=epsilon_schedule,
        batch_size=batch_size,
        horizon_kappa=horizon_kappa,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    logger = None
    if diagnostics:
        logger = DiagnosticsLogger()
        trainer.diag_logger = logger

    trainer.train()
    checkpoint_path = None
    diagnostics_summary = {}

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "delphos_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        if logger is not None:
            task_name_map = {int(task.id): str(task.name) for task in tasks}
            export_all_csvs(logger=logger, run_dir=checkpoint_dir, buffer=replay_buffer, task_name_map=task_name_map)
            diagnostics_summary = summarize_training(logger=logger, run_dir=checkpoint_dir, buffer=replay_buffer, task_name_map=task_name_map)

    _path = (None if checkpoint_path is None else str(checkpoint_path))
    return TrainingExperimentResult(trainer=trainer, checkpoint_path= _path , diagnostics_summary=diagnostics_summary,)