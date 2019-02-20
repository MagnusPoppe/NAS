import os
import sys
import src.laboratory.common as fn

parameter_error = """
This script needs the following parameters:
\t1. Path to configuration JSON file
\t2. Path to Module save directory containing genotype, phenotype (keras model) and (optional: image)
"""

if __name__ == '__main__':
    if len(sys.argv) == 3:
        config = fn.load_json_config(sys.argv[1])
        model_path = sys.argv[2]
    else:
        raise IOError(parameter_error)

    module = fn.load_module(os.path.join(model_path, "genotype.obj"))
    print(module.report)

