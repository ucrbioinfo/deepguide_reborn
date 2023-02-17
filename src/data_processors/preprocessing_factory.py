import argparse

from data_processors.preprocessing_base import PreprocessingBase
from data_processors.deepguide_one_preprocessing import DeepGuideOnePreprocessing
from data_processors.deepguide_two_preprocessing import DeepGuideTwoPreprocessing


class PreProcessingFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args


    def get_preprocessor(self, model_name: str) -> PreprocessingBase:
        match model_name:
            case 'dg1':
                return DeepGuideOnePreprocessing(self.args)
            case 'dg2':
                return DeepGuideTwoPreprocessing(self.args)
            case _:
                print('No model called {model}. Please select a valid model in config.yaml.'.format(
                    model=model_name,
                ))
                raise NotImplementedError