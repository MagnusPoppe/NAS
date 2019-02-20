import pandas as pd
from src.buildingblocks.module import Module

def load_json_config(filepath:str) -> dict:
    import json
    with open(filepath, "r") as f:
        config = json.load(f)
    return config


def load_module(filepath:str) -> Module:
    import pickle
    with open(filepath, "rb") as f:
        module = pickle.load(f)
    return module


def progress_report(report: dict) -> pd.DataFrame:
    from copy import deepcopy
    cifar_keys = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Frog", "Horse", "Ship", "Truck"]
    new_report = {}
    for key, value in enumerate(report.items()):
        i = int(key) # Numbered...
        new_report["({}) {}".format(key, cifar_keys[i])] = value

    return pd.DataFrame.from_dict(new_report)


