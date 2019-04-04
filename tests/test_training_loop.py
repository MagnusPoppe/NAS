import pickle
import unittest
import os

from src.configuration import Configuration
from src.jobs.TF_launcher import apply_training_results
from src.output import generation_finished
from src.pattern_nets import recombine
from src.pattern_nets.evaluation import apply_results
from src.pattern_nets.initialization import initialize_patterns
from src.training import prepare_training as trainer


class TestSyncTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a network:
        cls.patterns = initialize_patterns(count=5)
        nets = recombine.combine(cls.patterns, num_nets=1, min_size=3, max_size=5)

        # Parameters:
        cls.epochs = 1
        server_id = 0
        dev_id = 0
        job_id = 0

        # Setting necessary properties:
        cls.net = nets[0]
        cls.config = Configuration.from_json("./tests/fixtures/config.json")
        cls.config.type = "PatternNets"
        cls.training_args = (
            pickle.dumps(cls.net),
            pickle.dumps(cls.config),
            cls.epochs,
            server_id,
            dev_id,
            job_id
        )

    def test_1_run_training(self):
        self.assertIsNotNone(self.net)
        self.assertIsNotNone(self.config)
        res_dict = trainer.run(self.training_args)

        apply_training_results([self.net], [res_dict])
        apply_results(self.patterns, [self.net])
        current_patterns = [p for p in self.patterns if any(p.ID == q.ID for q in self.net.patterns)]
        self.assertTrue(
            all(p.results for p in current_patterns),
            "Missing results... "
        )
        generation_finished([self.net], "Test complete. Training stats:")

        epoch_stopped = self.net.epochs_trained

        res_dict2 = trainer.run(self.training_args)
        apply_training_results([self.net], [res_dict2])
        apply_results(self.patterns, [self.net])

        prev = self.net.validation_fitness[epoch_stopped-1]
        for i in range(epoch_stopped, epoch_stopped+5):
            self.assertTrue(prev * 0.975 <= self.net.validation_fitness[i] <= prev * 1.025)
