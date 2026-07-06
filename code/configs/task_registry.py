from mdp.task import Task
from pathlib import Path
from configs.task_configuration import PROJECT_ROOT

# List of task configurations intended for multi-task RL training
TASKS: list[Task] = [
    Task.from_yaml(0, PROJECT_ROOT.parent / "dataset/dataset_1/dataset.yaml"), # ApolloModeChoice
    Task.from_yaml(1, PROJECT_ROOT.parent / "dataset/dataset_2/dataset.yaml"), # ApolloRouteChoice
    Task.from_yaml(2, PROJECT_ROOT.parent / "dataset/dataset_3/dataset.yaml"), # Decisions
    Task.from_yaml(3, PROJECT_ROOT.parent / "dataset/dataset_5/dataset.yaml"), # 1987_NL_VoT
    Task.from_yaml(4, PROJECT_ROOT.parent / "dataset/dataset_6/dataset.yaml"), # 2009_Norway_VoT
    Task.from_yaml(5, PROJECT_ROOT.parent / "dataset/dataset_7/dataset.yaml"), # 2013_Arentze
    Task.from_yaml(6, PROJECT_ROOT.parent / "dataset/dataset_8/dataset.yaml"), # 2014_spain_parkingChoice
    Task.from_yaml(7, PROJECT_ROOT.parent / "dataset/dataset_9/dataset.yaml"), # 2018 LPMC
    Task.from_yaml(8, PROJECT_ROOT.parent / "dataset/dataset_10/dataset.yaml"), # 2018_Optima
    Task.from_yaml(9, PROJECT_ROOT.parent / "dataset/dataset_11/dataset.yaml"), # 2019_vanCranenburgh      
]

# List of task configurations reserved for zero-shot and few-shot transfer evaluation
INFERENCE_TASKS: list[Task] = [
    Task.from_yaml(3, PROJECT_ROOT.parent / "dataset/dataset_4/dataset.yaml"), # 2001_Swissmetro
]
