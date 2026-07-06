# Delphos Tutorials

These tutorials are written for final users who want to use Delphos for choice modelling inference and Apollo/R estimation.

## Recommended order

1. `01_getting_started.ipynb` - load Delphos, propose models, inspect Apollo specifications.
2. `02_datasets_and_user_data.ipynb` - understand bundled datasets and prepare your own CSV/schema.
3. `03_modelling_space.ipynb` - restrict search to transformations, tastes, covariates, attributes, and covariate levels.
4. `04_delphos_parameters.ipynb` - tune search strategy, exploration, depth, seeds, checkpoints, and estimation flags.
5. `05_reward_function.ipynb` - understand the built-in reward and build project-specific ranking scores.
6. `06_quick_results.ipynb` - run a compact workflow intended for first results within about 20 minutes.
7. `07_robust_results.ipynb` - structure long, resumable runs for robust and best-model search.
8. `08_environment_and_outputs.ipynb` - work directly with the Apollo/R environment, cache, saved output, and debugging.

## Safety defaults

Notebooks that call Apollo/R use flags such as `RUN_ESTIMATION = False` or `RUN_LONG = False` by default. Change those flags when you are ready to run estimation.

## Package imports

```
    cd /Users/gnova/Developer/Delphos
    pyenv local 3.12.11/envs/Delphos
    python -m pip install -r requirements.txt
    python -m pip install -e .
```

All tutorials use the final package API:

```python
import delphos as dp
agent = dp.load_agent()
dataset = dp.load_dataset("Swissmetro")
models = agent.propose(dataset, n_models=10)
```
