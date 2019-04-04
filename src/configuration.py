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
            address: str = "localhost",
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
        if self.type == "remote":
            import os
            if not os.path.exists(self.cwd):
                raise Exception("Work directory (cwd) does not exist...")


class Training(ValidatedInput):

    def __init__(self, epochs: float, batch_size: int, fixed_epochs: bool, use_restart: bool):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.fixed_epochs = fixed_epochs
        self.use_restart = use_restart


class Configuration(ValidatedInput):

    def __init__(
            self,
            dataset_name: str,
            dataset_file_path: str,
            dataset_file_name: str,
            input_format: (int,),
            classes_in_classifier: int,
            population_size: int,
            generations: int,
            results_name: str,
            save_all_results: bool,
            results_location: str,
            initial_min_network_size: int,
            initial_max_network_size: int,
            training: Training,
            servers: [Server]
    ):
        super().__init__()
        # Dataset Properties
        self.dataset_name = dataset_name

        # Training Properties
        self.training = training
        self.dataset_file_path = dataset_file_path
        self.dataset_file_name = dataset_file_name

        # Network Properties
        self.input_format = input_format
        self.classes_in_classifier = classes_in_classifier
        self.min_size = initial_min_network_size
        self.max_size = initial_max_network_size

        # EA Properties
        self.generation = 0
        self.generations = generations
        self.population_size = population_size

        # Results properties
        self.results_name = results_name
        self.save_all_results = save_all_results
        self.results_location = results_location

        # Compute Environment:
        self.MPI = False
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
        training = Training(
            batch_size=conf["training"]['batch size'],
            epochs=conf["training"]['epochs'],
            fixed_epochs=conf["training"]['fixed epochs'],
            use_restart=conf["training"]['use restart'],
        )

        return Configuration(
            dataset_name=conf['dataset'],
            dataset_file_path=conf['dataset path'],
            dataset_file_name=conf['dataset name'],
            input_format=tuple(conf['input']),
            classes_in_classifier=conf['classes'],
            population_size=conf['population size'],
            generations=conf['generations'],
            results_name=conf['run id'],
            save_all_results=conf['keep all results'],
            results_location=conf["results location"] if "results location" in conf else None,
            initial_min_network_size=conf['initial network size']['min'],
            initial_max_network_size=conf['initial network size']['max'],
            training=training,
            servers=servers
        )
