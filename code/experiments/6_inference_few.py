from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd
import torch
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdp.task import Task
from mdp.runtime import build_runtimes
from agent.delphos import Delphos
from agent.delphos_dqn import DelphosDQN
from agent.replay_buffer import make_replay_buffer
from agent.encoder import ZStateConfig
from trainer.trainer import Trainer
from mdp.state import Specification
from mdp.action import ActionSpace
from inference.inference_few_shot_learning import run_few_shot_inference

print(f"\nROOT: {ROOT}")

# =====================================================
# REPRODUCIBILITY SEED
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Global random seed set to: {SEED} for reproducibility")

# =====================================================
# Tasks
# =====================================================
task_1  = Task.from_yaml(0, yaml_path="./dataset/dataset_1/dataset.yaml")
task_2  = Task.from_yaml(1, yaml_path="./dataset/dataset_2/dataset.yaml")
task_3  = Task.from_yaml(2, yaml_path="./dataset/dataset_3/dataset.yaml") # Decisions
task_4  = Task.from_yaml(3, yaml_path="./dataset/dataset_4/dataset.yaml") # Swissmetro
task_5  = Task.from_yaml(4, yaml_path="./dataset/dataset_5/dataset.yaml")
task_6  = Task.from_yaml(5, yaml_path="./dataset/dataset_6/dataset.yaml")
task_7  = Task.from_yaml(6, yaml_path="./dataset/dataset_7/dataset.yaml")
task_8  = Task.from_yaml(7, yaml_path="./dataset/dataset_8/dataset.yaml")
task_9  = Task.from_yaml(8, yaml_path="./dataset/dataset_9/dataset.yaml")
task_10 = Task.from_yaml(9, yaml_path="./dataset/dataset_10/dataset.yaml")
task_11 = Task.from_yaml(10, yaml_path="./dataset/dataset_11/dataset.yaml")

all_tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10, task_11]

# Target inference tasks (Decisions and Swissmetro)
inference_tasks = [task_3, task_4]
linear_additive = True

# Create shared catalogue and action space based on all tasks
_, catalogue = build_runtimes(all_tasks, linear_additive, device="cpu")

specification_manager = Specification(catalogue=catalogue)
action_space = ActionSpace(task_1, catalogue, specification_manager, linear_additive=linear_additive)

def create_delphos_agent():
    return Delphos(
        num_actions=action_space.n_actions,
        z_cfg=ZStateConfig(
            K=catalogue.n_attributes,
            T=catalogue.n_transformations,
            G=catalogue.n_tastes,
            C=catalogue.n_covariates,
        )
    )

def create_dqn_agent():
    return DelphosDQN(
        num_actions=action_space.n_actions,
        z_cfg=ZStateConfig(
            K=catalogue.n_attributes,
            T=catalogue.n_transformations,
            G=catalogue.n_tastes,
            C=catalogue.n_covariates,
            head_flag=False,
        ),
        encoder_kind="onehot"
    )

checkpoints_base = ROOT.parent / "checkpoints"

# Define the evaluation strategies
strategies = [
    {"name": "greedy", "epsilon": 0.0, "temperature": 1.0, "top_k": 5},
    {"name": "stochastic", "epsilon": 0.05, "temperature": 1.0, "top_k": 5},
    {"name": "topk", "epsilon": 0.0, "temperature": 1.0, "top_k": 5},
    {"name": "boltzmann", "epsilon": 0.0, "temperature": 1.0, "top_k": 5}
]

adaptation_modes = ["policy_only", "encoder_only", "fine_tune_all"]

seeds = [123 + i for i in range(10)]

for seed in seeds:
    print(f"\n{'='*50}")
    print(f"Starting Few-Shot Inference for Trained Agent SEED: {seed}")
    print(f"{'='*50}\n")
   
    # Locate all possible checkpoint directories for this seed
    possible_checkpoint_names = []
    
    # DeepSet-Q Full
    possible_checkpoint_names.append(f"episode_full_agent_task_9_seed_{seed}")
    
    # DeepSet-Q Single & Seq & Base Full
    for t in range(1, 11):
        possible_checkpoint_names.append(f"single_agent_task_{t}_seed_{seed}")
        possible_checkpoint_names.append(f"episode_full_agent_task_{t}_seed_{seed}")
        possible_checkpoint_names.append(f"full_agent_task_{t}_seed_{seed}")
    
    # DQN Full
    possible_checkpoint_names.append(f"dqn_full_agent_task_9_seed_{seed}")
    
    # DQN Single
    for t in range(1, 10):
        possible_checkpoint_names.append(f"dqn_single_agent_task_{t}_seed_{seed}")

    for cp_name in possible_checkpoint_names:
        cp_dir = checkpoints_base / cp_name
        latest_cp = cp_dir / "latest_checkpoint.pt"

        if not latest_cp.exists():
            continue
            
        print(f"\nFound checkpoint: {cp_name}. Running Few-Shot Inference...")
        
        # Instantiate correct agent architecture
        if "dqn_" in cp_name:
            agent = create_dqn_agent()
        else:
            agent = create_delphos_agent()
            
        # Initialize dummy trainer
        dummy_buffer = make_replay_buffer(100, 'uniform')
        trainer = Trainer(
            agent=agent,
            runtimes={},
            catalogue=catalogue,
            linear_additive=linear_additive,
            replay_buffer=dummy_buffer,
            batch_size=64,
            horizon_kappa=2.0,
            seed=SEED, # Ensure trainer behavior is deterministic
            checkpoint_dir=None,
        )
        
        # Load the weights
        try:
            trainer.load_checkpoint(latest_cp)
            print(f"Successfully loaded checkpoint: {latest_cp}")
        except Exception as e:
            print(f"Error loading checkpoint {latest_cp}: {e}")
            continue
            
        # Run inference for each unseen task
        for task in inference_tasks:
            print(f"  -> Testing on {task.name}")
            
            output_dir = cp_dir / "inference_results" / f"task_{task.id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            task_results = []
            
            for adapt_mode in adaptation_modes:
                print(f"     -> Adaptation Mode: {adapt_mode}")
                for strat in strategies:
                    strat_name = strat["name"]
                    print(f"        -> Strategy: {strat_name}")
                    
                    try:
                        result_dict = run_few_shot_inference(
                            trainer=trainer,
                            task=task,
                            adaptation_mode=adapt_mode,
                            search_strategy=strat_name,
                            adaptation_rounds=50, # Standard default
                            epsilon=strat["epsilon"],
                            temperature=strat["temperature"],
                            top_k=strat["top_k"],
                            overwrite=True
                        )
                        task_results.append(result_dict)
                        
                    except Exception as e:
                        print(f"        Error running few-shot on {task.name} with {adapt_mode}_{strat_name}: {e}")

            if task_results:
                df = pd.DataFrame(task_results)
                out_csv = output_dir / "few_shot_results.csv"
                df.to_csv(out_csv, index=False)
                print(f"     Results saved to {out_csv} ({len(df)} configurations tested)")
