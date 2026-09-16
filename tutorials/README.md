# Delphos Tutorials

These tutorials are written for final users who want to use Delphos for choice modelling inference and Apollo/R estimation.

## Recommended order

0. `00_installation.ipynb` - verify your pip installation and R configuration.
1. `01_getting_started.ipynb` - load Delphos, propose models, inspect Apollo specifications.
2. `02_your_own_datasets.ipynb` - understand bundled datasets and prepare your own CSV/schema.
3. `03_modelling_space.ipynb` - restrict search to transformations, tastes, covariates, attributes, and covariate levels.
4. `04_advanced_search.ipynb` - tune search strategy with schedules, loop long robust runs, and leverage the ResultCache.
5. `05_reward_function.ipynb` - understand the built-in reward and build project-specific ranking scores (AIC, BIC, adjRho2).
6. `06_exploring_in_r.ipynb` - extract `.R` scripts from generated proposals and run them natively in RStudio.

## Safety defaults

Notebooks that call Apollo/R use flags such as `RUN_ESTIMATION = False` or `RUN_LONG = False` by default. Change those flags when you are ready to run estimation. Note that `rpy2` is no longer required! Delphos seamlessly runs `.R` scripts as independent subprocesses.

## Package imports

```bash
cd Delphos
python -m pip install -e .
python -m pip install jupyterlab
jupyter lab tutorials
```

All tutorials use the final package API:

```python
import delphos as dp
agent = dp.load_agent()
dataset = dp.load_dataset("dataset_4")
models = agent.propose(dataset, n_models=10)
```
