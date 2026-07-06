from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdp.task import Task
from mdp.runtime import build_runtimes

from agent.delphos import Delphos
from agent.replay_buffer import make_replay_buffer
from agent.encoder import ZStateConfig
from trainer.trainer import Trainer
from trainer.episode import run_episode
from mdp.runtime import evaluate_specification
from mdp.state import Specification
from mdp.action import *
from env.apollo.schema import ApolloSpecification
from env.apollo.generator import ApolloGenerator
from env.environment import evaluate_specification

print(f"\nROOT: {ROOT}")
# =====================================================
# Tasks
# =====================================================
task_1  = Task.from_yaml(0, yaml_path="./dataset/dataset_1/dataset.yaml")
task_2  = Task.from_yaml(1, yaml_path="./dataset/dataset_2/dataset.yaml")
task_3  = Task.from_yaml(2, yaml_path="./dataset/dataset_3/dataset.yaml")
task_4  = Task.from_yaml(3, yaml_path="./dataset/dataset_5/dataset.yaml")
task_5  = Task.from_yaml(4, yaml_path="./dataset/dataset_6/dataset.yaml")
task_6  = Task.from_yaml(5, yaml_path="./dataset/dataset_7/dataset.yaml")
task_7  = Task.from_yaml(6, yaml_path="./dataset/dataset_8/dataset.yaml")
task_8  = Task.from_yaml(7, yaml_path="./dataset/dataset_9/dataset.yaml")
task_9  = Task.from_yaml(8, yaml_path="./dataset/dataset_10/dataset.yaml")
task_10 = Task.from_yaml(9, yaml_path="./dataset/dataset_11/dataset.yaml")

tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10]#, 
linear_additive = True

runtimes, catalogue = build_runtimes(tasks, linear_additive, device="cpu")

specification_manager = Specification(catalogue=catalogue)
action_space = ActionSpace(task_1, catalogue, specification_manager,linear_additive=linear_additive)

# =====================================================
# Replay Buffer
# =====================================================
replay_buffer = make_replay_buffer(10_000, 'uniform')

# =====================================================
# Agent
# =====================================================

import random
import numpy as np
import torch

SEED = 123
#SEED = 123 + 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

agent = Delphos(
    num_actions = action_space.n_actions,
    z_cfg = ZStateConfig(
    K=catalogue.n_attributes,
    T=catalogue.n_transformations,
    G=catalogue.n_tastes,
    C=catalogue.n_covariates,
)    
)

# =====================================================
# Trainer
# =====================================================
conf_name = "task_10"
conf_dir = "checkpoints/" + conf_name
conf_freq = 5

trainer = Trainer(
    agent=agent,
    runtimes=runtimes,
    catalogue=catalogue,
    linear_additive = linear_additive,
    replay_buffer=replay_buffer,
    batch_size=64,
    horizon_kappa=2.0,
    seed=SEED,
    checkpoint_dir=conf_dir,
    checkpoint_frequency=conf_freq,
)

print("\nTrainer")
print(trainer.summary())


# =====================================================
# Single episode sanity check
# =====================================================
for i in range(len(runtimes)):
    print(f"\nRuntime {i}")

    print(runtimes[i].task.rewards_path)    



# =====================================================
# Training
# =====================================================

print("\nStarting training...\n")
trainer.train()

print("\nTraining finished")

print("Episodes:", trainer.episode_count)
print("Updates :", trainer.learn_steps)
print("Replay size:", len(trainer.replay_buffer))

# =====================================================
# Final greedy evaluation
# =====================================================

agent.inference_mode()

runtime = trainer.get_runtime(0)
final_episode = trainer.generate_episode(runtime=runtime,epsilon=0.0,)

print("\nFinal greedy evaluation")
print("Reward:", final_episode.reward)
print("Length:", final_episode.episode_length)
print("Specification:", final_episode.specification_key)