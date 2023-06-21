import sys
import yaml
import argparse


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)

    help = '''
    - The config file to use. Must be placed in the root folder.
    '''
    config_parser.add_argument(
        '-c',
        '--config',
        type=argparse.FileType(mode='r'),
        default='config.yaml',
        help=help,
    )
    
    config_args, remaining_args = config_parser.parse_known_args()

    config_arg_dict = vars(config_args)
    if config_args.config:
        config_arg_dict.update(yaml.load(config_args.config, Loader=yaml.FullLoader))

    parser = argparse.ArgumentParser(
        prog='DeepGuide',
        description='Deep learning based Guide RNA scoring framework.',
        parents=[config_parser],
    )

    help = '''
    - Name of the experiment. Output directory and file will be labeled with this.
    '''
    parser.add_argument(
        '-n',
        '--experiment_name',
        type=str,
        help=help
    )

    help = '''
    - Which model to use? Default: 'dg1'
    - Options are 'dg1'
    '''
    parser.add_argument(
        '--model',
        type=str,
        default='dg1',
        help=help,
    )

    help = '''
    - Which mode to use?
    - Options are 'pretrain', 'train', 'inference', 'pt' (pretrain + train), 'pti' (pretrain + train + inference), 'ti' (train + inference)
    '''
    parser.add_argument(
        '--mode',
        type=str,
        help=help,
    )

    help = '''
    - Which cas9 mode to use?
    - Options are 'cas9_seq'
    '''
    parser.add_argument(
        '--cas',
        type=str,
        help=help,
        default='cas9_seq',
    )

    help = '''
    - Length of the input guide RNA. Default: 28
    '''
    parser.add_argument(
        '--guide_length',
        type=int,
        default=1,
        help=help,
    )

    help = '''
    - Input directory.
    '''
    parser.add_argument(
        '--input_directory',
        type=str,
        help=help,
        default='data/input/',
    )

    help = '''
    - Output directory.
    '''
    parser.add_argument(
        '--output_directory',
        type=str,
        help=help,
        default='data/output/',
    )

    help = '''
    - The name of the entire genome file if using pretraining.
    - Place this in the input directory.
    '''
    parser.add_argument(
        '--pretrain_genome_input_name',
        type=str,
        help=help,
        default='',
    )

    help = '''
    - Name of the input csv file that contains the guide RNA to train on.
    '''
    parser.add_argument(
        '--train_guides_csv_file_name',
        type=str,
        help=help,
        default='example_train.csv',
    )

    help = '''
    - Column name which contains the guide RNA to train on in the input csv file.
    '''
    parser.add_argument(
        '--train_guide_seq_col_name',
        type=str,
        help=help,
    )

    help = '''
    - Column name which contains the guide RNA scores to train on in the input csv file.
    '''
    parser.add_argument(
        '--train_guide_score_col_name',
        type=str,
        help=help,
    )

    help = '''
    - Column name which contains the guide RNA to test on in the input csv file.
    '''
    parser.add_argument(
        '--inference_guides_csv_file_name',
        type=str,
        help=help,
    )

    help = '''
    - Column name which contains the guide RNA scores to test on in the input csv file.
    '''
    parser.add_argument(
        '--inference_guide_seq_col_name',
        type=str,
        help=help,
    )

    help = '''
    - Output a diagram that shows the ML architecture?
    - Requires Pydot, and Pydotplus -- Not required to run an experiment.
    - Feel free to disable (set to False) if you don't have those packages.
    '''
    parser.add_argument(
        '--plot_model',
        type=bool,
        default=False,
        help=help
    )

    help = '''
    - True creates a metrics.txt file containing Spearman and Pearson correlation coefficients.
    - These will be calcualted on the validation split.
    '''
    parser.add_argument(
        '--metrics',
        type=bool,
        default=False,
        help=help
    )

    help = '''
    - True creates a AUC curve.
    - These will be calcualted on the validation split.
    '''
    parser.add_argument(
        '--plot_auc_curve',
        type=bool,
        default=False,
        help=help
    )

    help = '''
    - Controls the pretraining/validation splits.
    - Default: 0.7
    - 0.7 Means 70% of data will be used for training and 30% for validation.
    '''
    parser.add_argument(
        '--dg_one_pretrain_train_test_ratio',
        type=float,
        default=0.7,
        help=help
    )

    help = '''
    - Controls the training/validation splits.
    - Default: 0.7
    - 0.7 Means 70% of data will be used for training and 30% for validation.
    '''
    parser.add_argument(
        '--dg_one_train_test_ratio',
        type=float,
        default=0.7,
        help=help
    )

    help = '''
    - Controls learning rate of the Adam optimizer for CAE DeepGuide (dg1).
    - Default: 0.001
    '''
    parser.add_argument(
        '--dg_one_adam_lr',
        type=float,
        default=0.003,
        help=help
    )

    parser.add_argument(
        '--dg_one_roc_curve_threshold',
        type=float,
        default=3,
    )

    help = '''
    - Controls the batch size for CAE DeepGuide (dg1).
    - Default: 64
    '''
    parser.add_argument(
        '--dg_one_batch_size',
        type=int,
        default=64,
        help=help
    )

    help = '''
    - Controls the epoch count for CAE DeepGuide (dg1).
    - Default: 10
    '''
    parser.add_argument(
        '--dg_one_epochs',
        type=int,
        default=10,
        help=help
    )

    parser.set_defaults(**config_arg_dict)
    args = parser.parse_args(remaining_args)

    return args


def check_and_summarize_args(args: argparse.Namespace) -> None:
    model_names = dict({
        'dg1': 'DeepGuide 1 (CAE).',
    })

    modes = dict({
        'pretrain': 'Pre-Train',
        'train': 'Train',
        'inference': 'Inference',
        'pt': 'Pre-Train + Train',
        'ti': 'Train + Inference',
        'pti': 'Pre-Train + Train + Inference',
    })

    if args.model not in model_names:
        print('No model named {model_name}. Please select a valid model in config.yaml.'.format(
                model_name=args.model,
        ))

        print('Valid models are: {keys}'.format(keys=list(model_names.keys())))
        sys.exit(1)

    if args.mode not in modes:
        print('No mode named {mode}. Please select a valid model in config.yaml.'.format(
                mode=args.mode,
        ))

        print('Valid modes are: {keys}'.format(keys=list(modes.keys())))
        sys.exit(1)
    
    print()
    print('DeepGuide Overview:')
    print('- Experiment name: {name}.'.format(name=args.experiment_name))
    print('- Model: {model}'.format(model=model_names[args.model]))
    print('- Running in {mode} mode.'.format(mode=modes[args.mode]))
    print('- Cas mode: {mode}.'.format(mode=args.cas))
    print('- Using guides with length {length}.'.format(length=args.guide_length))
    
    if args.mode == 'dg1':
        print('Pre-Train genome input path: {path}'.format(
            path=args.input_path + args.pretrain_genome_input_name
            )
        )

    print()