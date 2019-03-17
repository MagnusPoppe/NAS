import unittest
import os
os.chdir("..")

from src.buildingblocks.ops.convolution import Conv3x3, Conv5x5
from src.buildingblocks.ops.pooling import AvgPooling2x2
from src.evolutionary_operations.initialization import init_population

from src.buildingblocks.module import Module
from src.buildingblocks.ops.dense import DenseS, DenseM, DenseL
from src.evolutionary_operations import mutation_operators as mutate


class TestDropoutIncluded(unittest.TestCase):

    def test_dropout_is_placed_correctly_in_simple_assembled_network(self):
        from tensorflow import keras
        from src.frameworks.keras import module_to_model
        module = Module(name="TestDropout")
        l1, l2, l3, l4 = DenseL(), DenseM(), DenseS(), DenseM()

        # Building model like:
        # l1 ->  l2  -> l3
        #    \         /
        #     -> l4 ->
        module = mutate.append(module, l1)
        module = mutate.append(module, l2)
        module = mutate.append(module, l3)
        module = mutate.insert(module, first=l1, last=l3, op=l4, between=False)

        # Assemble and display:
        model = module_to_model(module, input_shape=[784], classes=10)
        keras.utils.plot_model(model, to_file="tests/output/TestDropoutSimple.png")

    def test_dropout_is_placed_correctly_in_complex_assembled_network(self):
        from tensorflow import keras
        from src.frameworks.keras import module_to_model
        module = Module(name="TestDropout")
        l1, l2, l3, l4, l5, l6 = Conv5x5(), DenseL(), Conv3x3(dropout=False), AvgPooling2x2(), DenseL(), Conv3x3(dropout=False)

        # Building model with form:
        #         ->  l5 ->
        #      /            \
        #      |---------->  l3 ->
        #      |                   \
        # -> l1 -------> l2 ------> l5 ->
        #       \                  /
        #         -----> l4 ----->
        module = mutate.append(module, l1)
        module = mutate.append(module, l2)
        module = mutate.append(module, l5)
        module = mutate.insert(module, first=l1, last=l5, op=l3, between=False)
        module = mutate.insert(module, first=l1, last=l5, op=l4, between=False)
        module = mutate.insert(module, first=l1, last=l3, op=l6, between=False)

        # Assemble and display:
        model = module_to_model(module, input_shape=[32, 32, 3], classes=10)
        keras.utils.plot_model(model, to_file="tests/output/TestDropoutComplex.png")

    def test_many_randomly_generated_networks_are_created_stress_test(self):
        import pickle
        import multiprocessing as mp
        in_shape = [32, 32, 3]
        many_modules = init_population(1000, in_shape, network_min_layers=2, network_max_layers=200)
        many_modules = [(pickle.dumps((m, in_shape, 10))) for m in many_modules]
        print(f"Running test on {mp.cpu_count()} cores")
        pool = mp.Pool()
        runner = pool.map_async(parallel_reciever, many_modules)
        pool.close()
        results = runner.get()
        self.assertTrue(all(results))

def parallel_reciever(args):
    import pickle
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
    from src.frameworks.keras import module_to_model
    module, in_shape, classes = pickle.loads(args)
    model = module_to_model(module, in_shape, classes)
    print("=", end="", flush=True)
    return True
