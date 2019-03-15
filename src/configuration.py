class ValidatedInput():
    def __init__(self):
        pass

    def validate(self):
        pass


class ComputeDevice(ValidatedInput):
    def __init__(self, device_str: str, allow_memory_growth: bool, memory_per_process: float, concurrency: int):
        super().__init__()
        self.device = device_str
        self.allow_memory_growth = allow_memory_growth
        self.memory_per_process = memory_per_process
        self.concurrency = concurrency
        self.validate()

    def validate(self):
        safe = self.device[0] == "/" and (
                (
                        self.device.strip("/").split(":")[0] == "device" and
                        self.device.strip("/").split(":")[1] in ["CPU", "GPU"]
                ) or (self.device.strip("/").split(":")[0] in ["CPU", "GPU"])
        ) and self.device.split(":")[-1].isnumeric()

        if not safe:
            raise Exception("Device string must be /<CPU, GPU>:<int>")

        if not (0 < self.memory_per_process <= 1):
            raise Exception("Memory per process must be between 0.0 and 1.0")


class Server(ValidatedInput):
    def __init__(
            self,
            name: str,
            type: str,
            cwd: str,
            address: str = "",
            devices: [ComputeDevice] = None,
            python: str = "python3"
    ):
        super().__init__()
        self.name = name
        self.type = type.lower()
        self.cwd = cwd
        self.address = address
        self.devices = devices if devices else []
        self.python = python
        self.validate()

    def validate(self):
        if self.type not in ["local", "remote"]:
            raise Exception("Server type must be \"local\" or \"remote\"")
        if self.type == "local":
            import os
            if not os.path.exists(self.cwd):
                raise Exception("Work directory (cwd) does not exist...")


class Configuration(ValidatedInput):

    def __init__(
            self,
            dataset_name: str,
            training_loop_path: str,
            input_format: (int,),
            classes_in_classifier: int,
            batch_size: int,
            epochs_per_layer: float,
            population_size: int,
            generations: int,
            results_name: str,
            save_all_results: bool,
            servers: [Server]
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.training_loop_path = training_loop_path
        self.input_format = input_format
        self.classes_in_classifier = classes_in_classifier
        self.batch_size = batch_size
        self.epochs_per_layer = epochs_per_layer
        self.population_size = population_size
        self.generations = generations
        self.results_name = results_name
        self.save_all_results = save_all_results
        self.servers = servers
        self.validate()

    def validate(self):
        pass

    @staticmethod
    def from_json(json_path):
        import json
        with open(json_path, "r") as f:
            conf = json.load(f)
        servers = []
        for server in conf['servers']:
            compute = []
            for dev in server['devices']:
                compute += [
                    ComputeDevice(
                        dev["device_str"],
                        dev["allow gpu memory growth"],
                        dev["memory per process"],
                        dev["concurrency"])
                ]
            servers += [Server(server['name'], server['type'], server['cwd'], server['address'], compute, server['python'])]

        return Configuration(
            conf['dataset'],
            conf['trainingFilepath'],
            tuple(conf['input']),
            conf['classes'],
            conf['batch size'],
            conf['epochs'],
            conf['population size'],
            conf['generations'],
            conf['run id'],
            conf['keep all results'],
            servers
        )
