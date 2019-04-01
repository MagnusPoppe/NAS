import pickle
import unittest
import os

from src.configuration import Configuration
from src.jobs.TF_launcher import apply_results
from src.output import generation_finished
from src.pattern_nets import recombine
from src.pattern_nets.initialization import initialize_patterns
from src.training import prepare_training as trainer


class TestSyncTraining(unittest.TestCase):

    def setUp(self):
        # Create a network:
        patterns = initialize_patterns(count=5)
        nets = recombine.combine(patterns, num_nets=1, min_size=3, max_size=5)

        # Parameters:
        self.epochs = 1
        server_id = 0
        dev_id = 0
        job_id = 0

        # Setting necessary properties:
        self.net = nets[0]
        self.config = Configuration.from_json("./tests/fixtures/config.json")
        self.config.type = "PatternNets"
        self.training_args = (
            pickle.dumps(self.net),
            pickle.dumps(self.config),
            self.epochs,
            server_id,
            dev_id,
            job_id
        )


    def test_run_training(self):
        self.assertIsNotNone(self.net)
        self.assertIsNotNone(self.config)
        res_dict = trainer.run(self.training_args)

        for key, value in res_dict.items():
            self.assertIsNotNone(value, f"Property {key} of result dict was None...")

        apply_results([self.net], [res_dict])
        generation_finished([self.net], "Test complete. Training stats:")
