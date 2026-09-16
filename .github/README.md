# Delphos Continuous Integration (CI)

This directory contains the configurations for automated testing on GitHub, known as Continuous Integration (CI).

## How it works
Every time a developer pushes code to the repository or opens a Pull Request, GitHub Actions will read the `.github/workflows/test.yaml` file and automatically spin up a temporary virtual machine. 

On this machine, it will:
1. Install Python (testing multiple versions like 3.10, 3.11, 3.12).
2. Install the `Delphos` package and its dependencies (`pip install -e ".[dev]"`).
3. Run the entire `pytest` suite.
4. Generate a code coverage report to see which lines of code were executed during tests.

## Why it's important for Delphos
As scientific software, we need mathematical guarantees that refactoring code does not silently alter the agent's behavior or break the grammar logic. By automating this, we ensure that **nobody can merge broken code** into the main repository.

Because we mocked `Rscript` execution in `tests/env/test_apollo_estimator.py`, these tests run in seconds without requiring R to be installed on the GitHub cloud runner.
