import os
import pickle

from src.buildingblocks.pattern import Pattern


class ValidatedInput():
    def __init__(self):
        pass

    def validate(self):
        pass


class ResultStore(ValidatedInput):

    def __init__(self, name: str, location: str, keep_all: bool = False, load: str = None):
        super().__init__()
        self.name = name
        self.location = location if location else os.path.join(os.getcwd(), "results")
        self.keep_all = keep_all
        self.individ_store = os.path.join(self.location, self.name, "individs")
        self.generations_store = os.path.join(self.location, self.name, "generations")
        self.load = load
        os.makedirs(self.individ_store, exist_ok=True)
        os.makedirs(self.generations_store, exist_ok=True)

    def transfer_and_load_population(self):
        # Checking load directory:
        if not all("genotype.obj" in os.listdir(os.path.join(self.load, x)) for x in os.listdir(self.load)):
            raise IOError("Incomplete generation was tried loaded...")

        import shutil
        # Load all results from previous:
        genotypes = []
        for dir in os.listdir(self.load):
            filepath = os.path.join(self.load, dir, "genotype.obj")
            with open(filepath, "rb") as f_ptr:
                genotype = pickle.load(f_ptr)
                if not isinstance(genotype, Pattern):
                    continue
                genotypes += [genotype]

            # Copy files from previous
            for content in os.listdir(os.path.join(self.load, dir)):
                src = os.path.join(self.load, dir, content)
                dst = os.path.join(self.ensure_individ_path(genotypes[-1]), content)
                shutil.copy2(src, dst)
        return genotypes

    def store_generation(self, population, generation: int):
        # Creating current generation directory:
        generation_dir = os.path.join(self.generations_store, str(generation))
        os.makedirs(generation_dir, exist_ok=True)

        # Storing each individ:
        for individ in population:
            # Finding save directory and saving genotype:
            directory = self.ensure_individ_path(individ)
            with open(os.path.join(directory, "genotype.obj"), "wb") as f_ptr:
                pickle.dump(individ, f_ptr)

            # Adding shortcut to generation directory
            self.create_shortcut(directory, os.path.join(generation_dir, f"'{individ.ID}'"))

    def create_shortcut(self, a: str, b: str):
        """ Creates a shortcut of a stored as b, where a and b are abspaths"""
        os.symlink(a, b, target_is_directory=True)

    def ensure_individ_path(self, individ):
        path = os.path.join(self.individ_store, individ.name, str(individ.version))
        os.makedirs(path, exist_ok=True)
        return path


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
                raise Exception(f"Work directory ({self.cwd}) does not exist...")


class Training(ValidatedInput):

    def __init__(self, epochs: float, batch_size: int, learning_rate: float, fixed_epochs: bool, use_restart: bool):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fixed_epochs = fixed_epochs
        self.use_restart = use_restart
        self.acceptable_scores = None


class Dataset(ValidatedInput):

    def __init__(self, dataset_name: str, dataset_file_path: str, dataset_file_name: str, accepted_accuracy: float,
                 input: [int]):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_file_path = dataset_file_path
        self.dataset_file_name = dataset_file_name
        self.accepted_accuracy = accepted_accuracy
        self.input = input


class Configuration(ValidatedInput):

    def __init__(
            self,
            target_dataset: Dataset,
            pretrain_dataset: Dataset,
            input_format: (int,),
            classes_in_classifier: int,
            population_size: int,
            generations: int,
            initial_min_network_size: int,
            initial_max_network_size: int,
            training: Training,
            servers: [Server],
            async_verbose: bool,
            result: ResultStore
    ):
        super().__init__()
        # Dataset Properties (Guides)
        self.target_dataset = target_dataset
        self.pretrain_dataset = pretrain_dataset

        # Training Properties (Actually used)
        self.training = training
        self.dataset_name = None
        self.dataset_file_path = None
        self.dataset_file_name = None

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
        self.results = result

        # Compute Environment:
        self.async_verbose = async_verbose
        self.MPI = False
        self.servers = servers
        self.validate()

    def validate(self):
        pass

    def compute_capacity(self, maximum=True):
        return sum([dev.concurrency if maximum else 1 for server in self.servers for dev in server.devices])

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

            servers += [
                Server(server['name'], server['type'], server['cwd'], server['address'], compute, server['python'])]
        training = Training(
            batch_size=conf["training"]['batch size'],
            epochs=conf["training"]['epochs'],
            learning_rate=conf["training"]['learning rate'],
            fixed_epochs=conf["training"]['fixed epochs'],
            use_restart=conf["training"]['use restart']
        )

        result = ResultStore(
            name=conf['results']['name'],
            keep_all=conf['results']['keep all'],
            location=conf['results']['location'] if "location" in conf['results'] else None,
            load=conf['results']['load'] if "load" in conf['results'] else ""
        )

        target_dataset = Dataset(
            dataset_name=conf['target dataset']['dataset'],
            dataset_file_name=conf['target dataset']['dataset name'],
            dataset_file_path=conf['target dataset']['dataset path'],
            accepted_accuracy=conf['target dataset']['accepted_accuracy'],
            input=tuple(conf['target dataset']['input'])
        )
        pretrain_dataset = Dataset(
            dataset_name=conf['pretrain dataset']['dataset'],
            dataset_file_name=conf['pretrain dataset']['dataset name'],
            dataset_file_path=conf['pretrain dataset']['dataset path'],
            accepted_accuracy=conf['pretrain dataset']['accepted_accuracy'],
            input=tuple(conf['pretrain dataset']['input'])
        ) if "pretrain dataset" in conf else None

        return Configuration(
            target_dataset=target_dataset,
            pretrain_dataset=pretrain_dataset,
            input_format=None,
            classes_in_classifier=conf['classes'],
            population_size=conf['population size'],
            generations=conf['generations'],
            initial_min_network_size=conf['initial network size']['min'],
            initial_max_network_size=conf['initial network size']['max'],
            training=training,
            servers=servers,
            result=result,
            async_verbose=conf['verbose'] if 'verbose' in conf else True
        )
