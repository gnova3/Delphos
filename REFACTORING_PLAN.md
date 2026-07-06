# Delphos final-user package refactor plan

Delphos should become a lightweight inference package for final users. The
package loads trained checkpoints, proposes model specifications, and can send
those specifications to the modelling environment for Apollo/R estimation.

## Package language

- `propose`: generate candidate model specifications from a trained Delphos
  checkpoint.
- `environment`: evaluate proposed specifications through the Apollo/R backend.
- `estimate`: user-facing flag that asks Delphos to call the environment.
- `dataset`: a bundled or user-provided choice dataset with a Delphos schema.

## Public API target

```python
import delphos as dp

agent = dp.load_agent()
dataset = dp.load_dataset("Swissmetro")

models = agent.propose(dataset, n_models=25)

models_with_results = agent.propose(
    dataset,
    n_models=25,
    estimate=True,
)
```

By default, Delphos returns proposed models only. When `estimate=True`, Delphos
passes each proposed model to `delphos.env` and returns Apollo modelling
results.

## Target layout

```text
delphos/
  __init__.py
  api.py

  agent/
    __init__.py
    checkpoint.py
    encoder.py
    inference_agent.py
    q_network.py

  data/
    __init__.py
    builder.py
    registry.py
    schema.py
    validation.py

  grammar/
    __init__.py
    actions.py
    catalogue.py
    runtime.py
    specification.py
    task.py

  inference/
    __init__.py
    proposal.py
    results.py
    search.py

  env/
    __init__.py
    environment.py
    result_cache.py
    reward.py
    apollo/
      __init__.py
      estimator.py
      generator.py
      r_env.py
      schema.py
```

Top-level folders stay user-facing:

```text
checkpoints/
dataset/
tutorials/
requirements.txt
requirements-r.txt
install_r_requirements.R
```

Tutorials will be rewritten after the package API is stable.

## Keep

- The active Delphos agent architecture from `agent/delphos.py`.
- The encoder and Q-network.
- Task, catalogue, specification, action-space, and runtime logic.
- The full `env` layer, including Apollo generator, estimator, reward, and
  result cache.
- Bundled datasets and the production checkpoint selected for final users.

## Remove or rewrite

- Training modules and replay buffers.
- Diagnostics and experiment scripts.
- Legacy configs with hardcoded `Delphos-core` paths.
- Old agent variants that import deleted modules.
- Notebook-only helpers.
- Generated files: `__pycache__`, `.DS_Store`, SQLite WAL/SHM sidecars, and
  Apollo `.rds` outputs.

## Refactor phases

1. Create the installable `delphos` package and move active modules into it.
2. Add a checkpoint loader that reconstructs the agent from checkpoint metadata
   and ignores trainer/optimizer state.
3. Replace `run_zero_shot_inference` with accessible proposal methods:
   `agent.propose(...)` and `delphos.propose_models(...)`.
4. Add `estimate=False` by default. When true, call `delphos.env.environment`.
5. Add dataset helpers: `list_datasets()`, `load_dataset(...)`, and
   `create_dataset(...)`.
6. Validate user datasets against the trained global Delphos catalogue before
   proposals begin.
7. Update top-level tutorials against the new API.
8. Add smoke tests for imports, checkpoint loading, bundled dataset loading,
   proposal generation, and Apollo estimation.
