import sys
import yaml
import argparse

from ml_models.ml_model_factory import MlModelFactory
from data_processors.preprocessing_factory import PreProcessingFactory 
from postprocessing_decorators.metrics_decorator import MetricsDecorator
from postprocessing_decorators.plots_and_graphs_decorator import PlotsAndGraphsDecorator


def parse_arguments() -> argparse.Namespace:
    '''
    Read the values in config.yaml
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', 
        type=argparse.FileType(mode='r'),
        default='config.yaml', 
        help='The config file to use. Default: Must be placed in the root folder.',
    )

    args = parser.parse_args()
    arg_dict = vars(args)
    if args.config:
        arg_dict.update(yaml.load(args.config, Loader=yaml.FullLoader))

    return args


def check_and_summarize_args(args: argparse.Namespace):
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
    print('-Experiment name: {name}.'.format(name=args.experiment_name))
    print('-Model: {model}'.format(model=model_names[args.model]))
    print('-Running in {mode} mode.'.format(mode=modes[args.mode]))
    print('-Cas mode: {mode}.'.format(mode=args.cas))
    print('-Using guides with length {length}.'.format(length=args.guide_length))
    
    if args.mode == 'dg1':
        print('Pre-Train genome input path: {path}'.format(
            path=args.input_path + args.pretrain_genome_input_name
            )
        )

    print()


def main() -> int:
    args = parse_arguments()
    check_and_summarize_args(args)

    preprocessor = PreProcessingFactory(args).get_preprocessor(model_name=args.model)

    stack = MlModelFactory(args=args).get_model(model_name=args.model)

    if args.metrics:
        stack = MetricsDecorator(decorated_ml_model=stack, args=args)
    
    if args.plot_auc_curve:
        stack = PlotsAndGraphsDecorator(decorated_ml_model=stack, args=args)

    match args.mode:
        case 'pretrain':
            stack.pretraining_data = preprocessor.preprocess_pretrain()

            stack.pretrain()

        case 'train':
            stack.training_data = preprocessor.preprocess_train()

            stack.train()

        case 'inference':
            stack.inference_data = preprocessor.preprocess_inference()

            stack.inference()

        case 'pt':
            stack.pretraining_data = preprocessor.preprocess_pretrain()
            stack.training_data = preprocessor.preprocess_train()

            stack.pretrain()
            stack.train()

        case 'ti':
            stack.training_data = preprocessor.preprocess_train()
            stack.inference_data = preprocessor.preprocess_inference()

            stack.train()
            stack.inference()

        case 'pti':
            stack.pretraining_data = preprocessor.preprocess_pretrain()
            stack.training_data = preprocessor.preprocess_train()
            stack.inference_data = preprocessor.preprocess_inference()


            stack.pretrain()
            stack.train()
            stack.inference()

        case _:
            print('Unknown mode selected. Check config.yaml.')
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())