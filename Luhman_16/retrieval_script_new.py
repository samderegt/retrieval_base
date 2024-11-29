import argparse
from retrieval_base.retrieval_setup import RetrievalSetup

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file', type=str, help='Name of configuration file', 
        )
    args = parser.parse_args()

    # Import input file as 'conf'
    conf_string = str(args.config_file).replace('.py', '').replace('/', '.')
    conf = __import__(conf_string, fromlist=[''])

    RetrievalSetup(conf)
