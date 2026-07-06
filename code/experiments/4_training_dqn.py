from pathlib import Path
import sys
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdp.task import Task
from mdp.runtime import build_runtimes, build_runtime
from agent.delphos_dqn import DelphosDQN
from agent.replay_buffer import make_replay_buffer
from agent.encoder import ZStateConfig
from trainer.trainer import Trainer
from mdp.state import Specification
from mdp.action import ActionSpace

print(f"\nROOT: {ROOT}")

# =====================================================
# Tasks
# =====================================================
task_1  = Task.from_yaml(0, yaml_path="./dataset/dataset_1/dataset.yaml")
task_2  = Task.from_yaml(1, yaml_path="./dataset/dataset_2/dataset.yaml")
task_3  = Task.from_yaml(2, yaml_path="./dataset/dataset_3/dataset.yaml")
task_4  = Task.from_yaml(3, yaml_path="./dataset/dataset_4/dataset.yaml")
task_5  = Task.from_yaml(4, yaml_path="./dataset/dataset_5/dataset.yaml")
task_6  = Task.from_yaml(5, yaml_path="./dataset/dataset_6/dataset.yaml")
task_7  = Task.from_yaml(6, yaml_path="./dataset/dataset_7/dataset.yaml")
task_8  = Task.from_yaml(7, yaml_path="./dataset/dataset_8/dataset.yaml")
task_9  = Task.from_yaml(8, yaml_path="./dataset/dataset_9/dataset.yaml")
task_10 = Task.from_yaml(9, yaml_path="./dataset/dataset_10/dataset.yaml")
task_11 = Task.from_yaml(10, yaml_path="./dataset/dataset_11/dataset.yaml")

all_tasks       = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10, task_11]
training_tasks  = [task_1, task_2, task_5, task_6, task_7, task_8, task_9, task_10, task_11]
inference_tasks = [task_3, task_4]
linear_additive = True

# Create shared catalogue and action space based on all tasks
runtimes_all, catalogue = build_runtimes(all_tasks, linear_additive, device="cpu")
runtimes_train, _ = build_runtimes(training_tasks, linear_additive, device="cpu")

specification_manager = Specification(catalogue=catalogue)
action_space = ActionSpace(task_1, catalogue, specification_manager, linear_additive=linear_additive)

# Setup 10 seeds
seeds = [123 + i for i in range(1, 10)]
conf_freq = 1000

for seed in seeds:
    print(f"\n{'='*50}")
    print(f"Starting DQN experiments for SEED: {seed}")
    print(f"{'='*50}\n")
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Function to create identical DQN agent instances
    def create_dqn_agent():
        return DelphosDQN(
            num_actions=action_space.n_actions,
            z_cfg=ZStateConfig(
                K=catalogue.n_attributes,
                T=catalogue.n_transformations,
                G=catalogue.n_tastes,
                C=catalogue.n_covariates,
                head_flag=False, # standard for one-hot is not to use a projection head
            ),
            encoder_kind="onehot"
        )

    # =====================================================
    # 1. Full Agent 
    # =====================================================
    print(f"\n--- Training Full DQN Agent (9 Tasks) - Seed {seed} ---")
    
    agent_full = create_dqn_agent()
    replay_buffer_full = make_replay_buffer(10_000, 'uniform')
    
    conf_name_full = f"dqn_full_agent_task_{len(training_tasks)}_seed_{seed}"
    conf_dir_full = f"checkpoints/{conf_name_full}"
    
    trainer_full = Trainer(
        agent=agent_full,
        runtimes=runtimes_train,
        catalogue=catalogue,
        linear_additive=linear_additive,
        replay_buffer=replay_buffer_full,
        batch_size=64,
        horizon_kappa=2.0,
        seed=seed,
        checkpoint_dir=conf_dir_full,
        checkpoint_frequency=conf_freq,
    )
    
    #trainer_full.train()       

    # =====================================================
    # 2. Single Task Agents
    # =====================================================
    print(f"\n--- Training Single Task DQN Agents - Seed {seed} ---")
    
    for t in range(1, len(training_tasks) + 1):
        aux_tasks = [training_tasks[t-1]] # Fixed indexing from original script
        
        aux_runtimes = {}
        for aux_t in aux_tasks:
            aux_runtimes[aux_t.id] = build_runtime(aux_t, catalogue, linear_additive, device='cpu')

        agent_single = create_dqn_agent()
        replay_buffer_single = make_replay_buffer(10_000, 'uniform')

        conf_name_single = f"dqn_single_agent_task_{t}_seed_{seed}"
        conf_dir_single = f"checkpoints/{conf_name_single}"

        trainer_single = Trainer(
            agent=agent_single,
            runtimes=aux_runtimes,
            catalogue=catalogue,
            linear_additive=linear_additive,
            replay_buffer=replay_buffer_single,
            batch_size=64,
            horizon_kappa=2.0,
            seed=seed,
            checkpoint_dir=conf_dir_single,
            checkpoint_frequency=conf_freq,
        )

        print(f"\nTraining Single DQN Agent for Task {t} (Seed {seed})...")
        # In this experiment we run the single task training.
        # It's recommended to test first.
        trainer_single.train()

    
