"""
trainer/trainer.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: Multi-task Delphos trainer.
"""


from __future__ import annotations

from pathlib import Path
from typing import Optional

import random
import time
import numpy as np
import torch
import json
from agent.delphos import Delphos
from agent.replay_buffer import UniformReplayBuffer
from mdp.runtime import build_runtime

from mdp.runtime import Runtime
from mdp.catalogue import Catalogue
from .episode import EpisodeResult, run_episode

from diagnostics.diagnostics_logger import DiagnosticsLogger
from diagnostics.diagnostics_summary import export_all_csvs

class Trainer:
    def __init__(self, agent: Delphos, runtimes: dict[int, Runtime], catalogue: Catalogue, linear_additive: bool,
                    replay_buffer: UniformReplayBuffer, epsilon_schedule: list[float] = None, 
                    batch_size: int = 64, horizon_kappa: float = 2.0, 
                    checkpoint_dir: Optional[str] = None, seed: Optional[int] = 123, checkpoint_frequency: int = 500):
        self.agent = agent
        self.runtimes = runtimes
        self.catalogue = catalogue
        self.linear_additive = linear_additive
        self.replay_buffer = replay_buffer
        self.batch_size = int(batch_size)
        self.horizon_kappa = float(horizon_kappa)
        self.epsilon_schedule = self.linear_epsilon_schedule() if epsilon_schedule is None else epsilon_schedule
        self.episode_count = 0
        self.learn_steps = 0
        self.current_epsilon = next(iter(self.epsilon_schedule))
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_frequency = int(checkpoint_frequency)
        self.seed = seed
        self.round_count = 0
        self.diag_logger = DiagnosticsLogger()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def summary(self) -> dict:
        return {
            "trainer": "Trainer",
            "episodes": self.episode_count,
            "learn_steps": self.learn_steps,
            "batch_size": self.batch_size,  
            "horizon_kappa": self.horizon_kappa,
            "num_tasks": len(self.runtimes),
            "checkpoint_dir": (None if self.checkpoint_dir is None else str(self.checkpoint_dir)),
            "agent": self.agent.summary(),
        }

    def generate_episode(self, runtime: Runtime, epsilon: float, boltzmann: bool = False, temperature: float = 1.0) -> EpisodeResult:
        _episode = run_episode(self.agent, runtime, epsilon, self.agent.discount_factor, self.horizon_kappa, boltzmann, temperature)
        return _episode
    
    def generate_episode_topk(self,runtime: Runtime,k: int = 5,temperature: float | None = None,) -> EpisodeResult:
        return run_episode(self.agent, runtime, temperature, self.agent.discount_factor, self.horizon_kappa, top_k=k, )

    def store_episode(self, episode: EpisodeResult) -> None:
        self.replay_buffer.add_episode(episode.transitions)

    def update_from_replay(self) -> Optional[dict]:
        sampled = self.replay_buffer.sample(self.batch_size)
        if sampled is None:
            return None
        batch, indices, is_w = sampled
        result = self.agent.update_from_batch(batch, indices, is_w, self.runtimes, self.replay_buffer)

        if result is not None:
            self.diag_logger.log_update(
                episode=self.episode_count,
                round_idx=self.round_count,
                epsilon=self.current_epsilon,
                loss=result["loss"],
                td_errors=result["td_errors"],
                q_values_sa=result["q_values_sa"],
                target_values=result["target_values"],
                grad_norm=result["grad_norm"],
                param_update_magnitude=result["param_update_magnitude"],
            )

        self.learn_steps += 1
        return result

    def run_round(self, epsilon: float) -> dict:
        rewards = []
        tasks = list(self.runtimes.values())
        random.shuffle(tasks)

        for runtime in tasks:
            episode = self.generate_episode(runtime, epsilon)
            if self.diag_logger is not None:
                self.diag_logger.log_episode(
                task_id=episode.task_id,
                task_name=episode.task_name,
                episode=self.episode_count,
                round_idx=self.round_count,
                epsilon=epsilon,
                reward=episode.reward,
                L_e=episode.episode_length,
                H_tau=episode.horizon,
                forced_stop=episode.forced_stop,
                spec_key=episode.specification_key,
                action_sequence=episode.action_indices,
                valid_actions_per_step=episode.valid_actions_per_step,
                unique_specifications=episode.unique_specifications,
                unique_actions=episode.unique_actions,
                runtime_seconds=episode.runtime_seconds,
                z_states=episode.z_states,
            )
            self.store_episode(episode)
            rewards.append(episode.reward)
            self.episode_count += 1
        update_result = self.update_from_replay()
        self.round_count += 1
        self.maybe_save_checkpoint()
        return {"mean_reward": float(np.mean(rewards)), "update": update_result}

    def run_episode(self, epsilon: float) -> dict:

        runtime = random.choice(list(self.runtimes.values()))
        episode = self.generate_episode(runtime, epsilon)
        if self.diag_logger is not None:
            self.diag_logger.log_episode(
                task_id=episode.task_id,
                task_name=episode.task_name,
                episode=self.episode_count,
                round_idx=self.round_count,
                epsilon=epsilon,
                reward=episode.reward,
                L_e=episode.episode_length,
                H_tau=episode.horizon,
                forced_stop=episode.forced_stop,
                spec_key=episode.specification_key,
                action_sequence=episode.action_indices,
                valid_actions_per_step=episode.valid_actions_per_step,
                unique_specifications=episode.unique_specifications,
                unique_actions=episode.unique_actions,
                runtime_seconds=episode.runtime_seconds,
                z_states=episode.z_states,
            )

        self.store_episode(episode)
        self.episode_count += 1
        update_result = self.update_from_replay()
        self.round_count += 1
        self.maybe_save_checkpoint()
        return {"mean_reward": float(episode.reward), "update": update_result}

    def get_runtime(self, task_id: int) -> Runtime:
        return self.runtimes[int(task_id)]

    def train_rounds(self, n_rounds: int, epsilon: float) -> None:
        for _ in range(int(n_rounds)):
            self.run_round(epsilon=epsilon)

    def train_episode(self, n_episodes: int, epsilon: float) -> None:
        for _ in range(int(n_episodes)):
            self.run_episode(epsilon)
            
    def train(self) -> None:        
        self.agent.full_training_mode()
        start_time = time.time()
        
        for epsilon in self.epsilon_schedule:
            self.current_epsilon = float(epsilon)
            self.run_round(self.current_epsilon)

        elapsed = time.time() - start_time
        print(f"Training finished | rounds={self.round_count} | episodes={self.episode_count} | updates={self.learn_steps} | time={elapsed:.1f}s")
        export_all_csvs(logger=self.diag_logger, run_dir=self.checkpoint_dir)

    def add_task(self, task, overwrite: bool = False) -> Runtime:      
        task_id = int(task.id)
        if task_id in self.runtimes and not overwrite:
            raise ValueError(f"Task id {task_id} already exists. Use overwrite=True to replace it.")

        runtime = build_runtime(task=task, catalogue=self.catalogue, linear_additive=self.linear_additive, device=str(self.agent.device))
        self.runtimes[task_id] = runtime   
        return runtime

    # =====================================================
    # Checkpointing
    # =====================================================
    def save_checkpoint(self, path: str | Path, save_mode: str = "full", extra_metadata: Optional[dict] = None) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "trainer": {
                "round_count": self.round_count,
                "episode_count": self.episode_count,
                "learn_steps": self.learn_steps,
                "current_epsilon": self.current_epsilon,
                "batch_size": self.batch_size,
                "horizon_kappa": self.horizon_kappa,
                "epsilon_schedule": self.epsilon_schedule,
                "linear_additive": self.linear_additive,
                "seed": self.seed,
            },
            "trainer_summary": self.summary(),
            "rng": {"numpy": np.random.get_state(), "torch": torch.get_rng_state(), "python": random.getstate(),},
        }
        agent_path = str(path) + ".agent.tmp"
        self.agent.save_checkpoint(agent_path, save_mode, extra_metadata)
        checkpoint["agent_checkpoint"] = torch.load(agent_path, map_location="cpu", weights_only=False)
        Path(agent_path).unlink(missing_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path, load_mode: str = "resume") -> dict:

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.agent.device, weights_only=False)

        if "trainer" not in checkpoint:
            raise ValueError("Checkpoint does not contain trainer state.")
        if "agent_checkpoint" not in checkpoint:
            raise ValueError("Checkpoint does not contain agent state.")

        rng_state = checkpoint.get("rng")
        if rng_state is not None:
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            random.setstate(rng_state["python"])

        trainer_state = checkpoint["trainer"]
        self.episode_count = int(trainer_state.get("episode_count", 0))
        self.learn_steps = int(trainer_state.get("learn_steps", 0))
        self.current_epsilon = float(trainer_state.get("current_epsilon", self.current_epsilon))
        tmp_agent_path = str(path) + ".agent.tmp"
     
        try:
            torch.save(checkpoint["agent_checkpoint"], tmp_agent_path)
            agent_metadata = self.agent.load_checkpoint(tmp_agent_path, load_mode=load_mode)
        finally:
            Path(tmp_agent_path).unlink(missing_ok=True)

        return {
            "trainer": trainer_state,
            "agent": agent_metadata,
            "trainer_summary": checkpoint.get("trainer_summary", {})
        }

    def maybe_save_checkpoint(self) -> None:
        if self.checkpoint_dir is None:
            return
        if self.episode_count == 0:
            return
        if self.round_count % self.checkpoint_frequency != 0:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        self.save_checkpoint(path=checkpoint_path, save_mode="full")
        print(f"Checkpoint saved (episode={self.episode_count}) -> {checkpoint_path}")

    def checkpoint_summary(self) -> dict:
        summary = {
            "trainer": self.summary(),
            "agent": self.agent.summary(),
            "tasks": {
                task_id: {
                    "id": runtime.task.id,
                    "name": runtime.task.name,
                    "yaml_path": str(runtime.task.yaml_path),
                    "dataset_path": str(runtime.task.dataset_path),
                    "rewards_path": str(runtime.task.rewards_path)
                } for task_id, runtime in self.runtimes.items()}}
        return summary

    def maybe_save_checkpoint(self) -> None:

        if self.checkpoint_dir is None:
            return
        if self.episode_count == 0:
            return
        if self.round_count % self.checkpoint_frequency != 0:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        summary_path = self.checkpoint_dir / "latest_summary.json"

        self.save_checkpoint(path=checkpoint_path, save_mode="full",)

        with open(summary_path, "w") as f:
            json.dump(self.checkpoint_summary(), f, indent=4,)

        print(f"Checkpoint saved (round={self.round_count}, episodes={self.episode_count}) -> {checkpoint_path}")

    @staticmethod
    def linear_epsilon_schedule(epsilon_start: float = 1.0, epsilon_end: float = 0.05, n_rounds: int = 10_000) -> list[float]:
        return np.linspace(epsilon_start, epsilon_end, n_rounds).tolist()

    