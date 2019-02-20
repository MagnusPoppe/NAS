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


def progress_report(module: Module):
    pass

