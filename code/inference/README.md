# Transfer module

The Delphos transfer module is responsible for evaluating pre-trained Delphos agents on unseen datasets and choice tasks. It provides a robust, multi-scenario pipeline designed to test both **Zero-Shot Inference** (how well the agent generalizes purely through exploration strategies) and **Few-Shot Adaptation** (how quickly the agent can fine-tune its Q-network to a novel task).

### Zero-Shot Scenarios

In zero-shot mode, the agent's architecture weights are strictly frozen. The agent generates sequences of modelling actions based solely on its pre-trained state representations.

1. **Greedy Search** (`greedy`)
   Always selects the action that maximizes the predicted Q-value at the current step. It provides a deterministic baseline for the agent's absolute best guess.
2. **Stochastic Search** (`stochastic`)
   Implements an $\epsilon$-greedy exploration policy. With probability $1 - \epsilon$, it acts greedily; with probability $\epsilon$, it takes a random valid action. Useful to quickly sample around the greedy path.
3. **Boltzmann Search** (`boltzmann`)
   Selects actions by sampling from a categorical distribution formed by applying a Softmax function over the Q-values, scaled by a `temperature` parameter. Higher temperatures flatten the probabilities (more exploration), while lower temperatures sharpen them (approaching greedy).
4. **Top-K Search** (`topk`)
   At every step, computes the top $K$ actions with the highest Q-values and samples uniformly among them.
5. **Beam Search** (`beam`)
   A classical tree-search algorithm. It maintains a set (beam) of the $W$ most promising partial specification trajectories at each step. At the end of the depth, it yields the highest scoring paths. Highly effective for deep specification mining but computationally expensive.

### Few-Shot Scenarios (Online Adaptation)

In few-shot mode, the agent is allowed to actively fine-tune its neural network online using the reward signals (e.g., AIC/BIC or Log-Likelihood) produced by the unseen task. The original agent is copied to prevent cross-contamination.

1. **Q-Head Only** (`q_head_only`)
   Only the final Q-policy network is updated via backpropagation, while the DeepSet encoder is frozen.
2. **Encoder Only** (`encoder_only`)
   Only the DeepSet encoder is updated via backpropagation, while the Q-policy network is frozen.
3. **Full Fine-tuning** (`full`)
   The entire agent architecture is un-frozen and fine-tuned jointly on the unseen dataset.

---

## Usage Guide

### 1. Load the pre-trained agent

```python
from transfer.transfer_utils import (
    load_pretrained_trainer,
    get_tasks_and_runtimes,
    run_zero_shot_inference,
    run_few_shot_inference,
    plot_pareto
)
from configs.task_registry import INFERENCE_TASKS

# 1. Load the pre-trained agent and metadata automatically
EXPERIMENT_DIR = "experiments/small_full_pipeline/iteration_20260526_220728"
trainer = load_pretrained_trainer(experiment_dir=EXPERIMENT_DIR)

# 2. Get the environments for the target transfer tasks
tasks_and_runtimes = get_tasks_and_runtimes(trainer=trainer, tasks=INFERENCE_TASKS)

# 3. Evaluate Zero-Shot & Few-Shot scenarios
for task, runtime in tasks_and_runtimes:

    # Run zero-shot heuristics (automatically merges LLout/nFreeParams)
    df_zero_shot = run_zero_shot_inference(
        trainer=trainer,
        runtime=runtime,
        task=task,
        n_episodes=20,
        modes=["greedy", "boltzmann", "beam"]
    )

    # Run few-shot fine-tuning
    df_few_shot = run_few_shot_inference(
        trainer=trainer,
        runtime=runtime,
        task=task,
        n_adaptation_episodes=50,
        modes=["q_head_only"]
    )

    # ... Aggregate results into df_results ...
```

### 2. Pareto Front Visualization

The transfer evaluation provides a non-dominated Pareto front over the explored specifications (minimizing both complexity/`nFreeParams` and LogLikelihood/`LLout`). We display all specifications generated during the transfer evaluation across all zero-shot and few-shot scenarios.

- Pareto-optimal models are highlighted with bold markers and connected via a dashed frontier line.
- Sub-optimal models are rendered with low opacity.

```python
# Generate the interactive Pareto front!
fig = plot_pareto(df_results)
fig.show()
```

### Visual Example
Here is a snapshot of the generated Pareto front. Notice the non-dominated specifications on the lower-left boundary connected by the dashed line, with different inference modes distinctly marked.

![Delphos Transfer Pareto Front](pareto_front.png)
