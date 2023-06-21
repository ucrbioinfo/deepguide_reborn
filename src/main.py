import sys

from ml_models.ml_model_factory import MlModelFactory
from data_processors.preprocessing_factory import PreProcessingFactory 
from postprocessing_decorators.metrics_decorator import MetricsDecorator
from postprocessing_decorators.plots_and_graphs_decorator import PlotsAndGraphsDecorator
from utils.configurator import parse_arguments, check_and_summarize_args


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