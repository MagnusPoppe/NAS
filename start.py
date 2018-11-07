import os
os.environ['EA_NAS_UPLOAD_TO_FIREBASE'] = '1'

import src.main as EA_NAS

if __name__ == '__main__':
    import sys, json
    if len(sys.argv) > 2:
        raise IOError("Program requires dataset config file.")
    if not os.path.isfile(sys.argv[1]):
        raise IOError("File {} does not exist!".format(sys.argv[1]))

    config_file = sys.argv[1]
    with open(file=config_file, mode="r") as js:
        config = json.load(js)
        config['input'] = tuple(config['input'])

    EA_NAS.run(config)