import importlib
import argparse
from retrieval_base.retrieval import RetrievalSetup, RetrievalRun
    
if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file', type=str, help='Name of configuration file', 
        )
    
    parser.add_argument('--setup', '-s', action='store_true')
    parser.add_argument('--run', '-r', action='store_true')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--evaluation', '-e', action='store_true')
    parser.add_argument('--profiling', action='store_true')
    args = parser.parse_args()

    # Import input file as 'conf'
    config_file = str(args.config_file)
    config = importlib.import_module(config_file.replace('.py', ''))
    setattr(config, 'config_file', config_file)

    if args.setup:
        ret = RetrievalSetup(config)
        import sys; sys.exit()

    ret = RetrievalRun(config, resume=(not args.restart), evaluation=args.evaluation)
        
    if args.run:
        ret.run()
    if args.evaluation:
        ret.run_evaluation()
    if args.profiling:
        ret.run_profiling()