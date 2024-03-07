import argparse

from ml_models.ml_model_base import MLModel
from ml_models.deepguide_one import DeepGuideOne

class MlModelFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args


    def get_model(self, model_name: str) -> MLModel:
        match model_name:
            case 'dg1':
                return DeepGuideOne(self.args)
            # case 'dg2':
            #     return DeepGuideTwo(self.args)
            case _:
                print('No model called {model}. Please select a valid model in config.yaml.'.format(
                    model=model_name,
                ))
                raise NotImplementedError