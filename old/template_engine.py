import os
import sys
import json
import numpy as np
import random


def build(folder:str) -> dict:
    if not os.path.exists(folder):
        raise FileNotFoundError("No such folder {0}".format(folder))
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    templates = {}
    for file in files:
        if ".json" in file:
            with open(os.path.join(folder, file), "r") as f:
                spec = json.load(f)

            spec["parameters"] = parse_parameters(spec["parameters"])
            templates[spec["name"]] = spec
    return templates


def parse_parameters(parameters: dict) -> dict:
    for parameter_name, datatype in parameters.items():
        if datatype == "int":
            parameters[parameter_name] = np.int
        elif datatype == "double":
            parameters[parameter_name] = np.double

    return parameters


def shuffle_parameters(operation):
    for param_name, param in operation["parameters"].items():
        if param["type"] in ["int", "float"]:
            if "max" in param: high = param["max"]
            else: high = sys.maxsize
            if "min" in param: low = param["min"]
            else: low = -sys.maxsize

            if param["type"] == "int":   param["value"] = random.randint(low, high) -1
            if param["type"] == "float": param["value"] = random.uniform(low, high)

        if param["type"] == "string":
            selected = random.randint(0, len(param["possibleValues"])) -1
            param["value"] = param["possibleValues"][selected]

        if param["type"] == "bool":
            param["value"] = (random.uniform(0, 1) <= param["probabilityTrue"])

    return operation