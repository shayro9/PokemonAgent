import os
import sys

def validate_args(args):
    """Validate command-line arguments.

    :param args: Parsed arguments.
    :raises SystemExit: If validation fails.
    """
    if args.warmup_data_path and not os.path.exists(args.warmup_data_path):
        print(f'ERROR: --warmup-data-path not found: {args.warmup_data_path}')
        sys.exit(1)

    if args.load_model_path:
        load_zip = args.load_model_path if args.load_model_path.endswith('.zip') else args.load_model_path + '.zip'
        if not os.path.exists(load_zip):
            print(f'ERROR: --load-model-path not found: {load_zip}')
            sys.exit(1)

    if args.eval_after_warmup and args.skip_eval:
        print('ERROR: --eval-after-warmup has no effect when --skip-eval is set.')
        sys.exit(1)

    if args.eval_after_warmup and args.eval_episodes == 0:
        print('ERROR: --eval-after-warmup requires --eval-episodes > 0.')
        sys.exit(1)
