# 00 - Installation and Setup

Welcome! Before you can run the rest of the tutorials, you need to ensure that the `Delphos` is installed.


### 1. R and Apollo prerequisites

Delphos estimates discrete choice model specifications using **Apollo in R**. Before installing `Delphos`, ensure that R and Apollo are installed and accessible on your system.

1. Install R from CRAN.

2. Install Apollo in R:

    ```r
    install.packages("apollo")
    ```

3. Add R to `PATH` (Windows)

   Delphos must be able to locate the R executable. On Windows, you may need to add the R installation directory to the system `PATH`.

   a. Open **Environment Variables** from the Windows Start menu.  
   b. Under **System variables**, select `Path` and click **Edit**.  
   c. Add the directory containing the R executable, for example:

   ```text
   C:\Program Files\R\R-x.y.z\bin\x64
   ```

   d. Save the changes and restart any open terminals, Anaconda, Conda, or Jupyter environments.

4. Verify that R is accessible from the command line:

    ```bash
    R --version
    ```


```python
!Rscript --version
```

    Rscript (R) version 4.5.1 (2025-06-13)


If the cell above printed the R version (e.g., `R scripting front-end version 4.x.x`), you are perfectly set up to run Delphos!

### 2. Installing Delphos

Install `Delphos` in editable mode:

```bash
    %pip install -e .
```

The `%pip` command installs `Delphos` in the Python environment associated with the current Jupyter kernel. 


```python
# Install the package located in the parent directory (Delphos root)
%pip install -e ..

```

    Obtaining file:///Users/gnova/Developer/Main-Delphos/Delphos
      Installing build dependencies ... [?25ldone
    [?25h  Checking if build backend supports build_editable ... [?25ldone
    [?25h  Getting requirements to build editable ... [?25ldone
    [?25h  Preparing editable metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: torch<2.9,>=2.0 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from delphos==0.1.0) (2.8.0)
    Requirement already satisfied: numpy<3.0,>=1.26 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from delphos==0.1.0) (2.5.3)
    Requirement already satisfied: pandas<3.0,>=2.2 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from delphos==0.1.0) (2.3.3)
    Requirement already satisfied: pyyaml<7.0,>=6.0 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from delphos==0.1.0) (6.0.3)
    Requirement already satisfied: tqdm<5.0,>=4.66 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from delphos==0.1.0) (4.70.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from pandas<3.0,>=2.2->delphos==0.1.0) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from pandas<3.0,>=2.2->delphos==0.1.0) (2026.3.post1)
    Requirement already satisfied: tzdata>=2022.7 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from pandas<3.0,>=2.2->delphos==0.1.0) (2026.4)
    Requirement already satisfied: filelock in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (3.32.7)
    Requirement already satisfied: typing-extensions>=4.10.0 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (4.16.0)
    Requirement already satisfied: setuptools in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (84.0.0)
    Requirement already satisfied: sympy>=1.13.3 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (1.14.0)
    Requirement already satisfied: networkx in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (3.6.1)
    Requirement already satisfied: jinja2 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (3.1.6)
    Requirement already satisfied: fsspec in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from torch<2.9,>=2.0->delphos==0.1.0) (2026.7.0)
    Requirement already satisfied: six>=1.5 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas<3.0,>=2.2->delphos==0.1.0) (1.17.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from sympy>=1.13.3->torch<2.9,>=2.0->delphos==0.1.0) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /Users/gnova/Developer/Main-Delphos/.venv/lib/python3.12/site-packages (from jinja2->torch<2.9,>=2.0->delphos==0.1.0) (3.0.3)
    Building wheels for collected packages: delphos
      Building editable for delphos (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for delphos: filename=delphos-0.1.0-0.editable-py3-none-any.whl size=1938 sha256=c3917382430dbc1d4b1261db4b5689c6b5a64e408ac5580024e117d14f9c9796
      Stored in directory: /private/var/folders/xv/k3gzdbk96pq928y4x8pxhsgsqf8dzx/T/pip-ephem-wheel-cache-8gvbhdpi/wheels/41/bc/73/8b899070f9498d2cd28c395d068b03e82aeacecf4d6c25044d
    Successfully built delphos
    Installing collected packages: delphos
      Attempting uninstall: delphos
        Found existing installation: delphos 0.1.0
        Uninstalling delphos-0.1.0:
          Successfully uninstalled delphos-0.1.0
    Successfully installed delphos-0.1.0
    Note: you may need to restart the kernel to use updated packages.


### 3. Verifying the Installation

Import `Delphos` and verify that the package, agent, and default model catalogue load correctly.


```python
import delphos as dp

print("Delphos successfully imported!")
```

    Delphos Backend: R version 4.5.1 (2025-06-13) | Apollo v0.3.7
    Delphos successfully imported!



```python
# Load the default agent
agent = dp.load_agent()
print("Agent loaded successfully! Device:", agent.agent.device)
```

    Agent loaded successfully! Device: cpu



```python
# Load the global catalogue
catalogue = dp.data.registry.load_global_catalogue()
print(f"Global catalogue loaded! It contains {catalogue.n_attributes} attributes and {catalogue.n_covariates} covariates.")

```

    Global catalogue loaded! It contains 7 attributes and 7 covariates.


You can now proceed to **`01_getting_started.ipynb`**.
