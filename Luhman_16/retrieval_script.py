import argparse
from retrieval_base.retrieval import pre_processing, Retrieval

import config_fiducial_K_A as conf
#import config_fiducial_K_B as conf
#import config_fiducial_J_A as conf
#import config_fiducial_J_B as conf

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_processing', action='store_true')
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    if args.pre_processing:
        pre_processing(conf=conf)

    if args.retrieval:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.PMN_run()

    if args.evaluation:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.PMN_callback_func(
            n_samples=None, 
            n_live=None, 
            n_params=None, 
            live_points=None, 
            posterior=None, 
            stats=None,
            max_ln_L=None, 
            ln_Z=None, 
            ln_Z_err=None, 
            nullcontext=None
            )

    if args.synthetic:
        ret = Retrieval(
            conf=conf, 
            evaluation=args.evaluation
            )
        ret.synthetic_spectrum()

