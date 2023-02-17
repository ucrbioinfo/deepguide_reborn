import pandas
import argparse
import tensorflow
from ml_models.ml_model_base import MLModel


class PostprocessingDecoratorBase(MLModel):
    def __init__(self, decorated_ml_model: MLModel, args: argparse.Namespace) -> None:
        self.decorated_ml_model = decorated_ml_model
        self.args = args
    
    
    def get_model(self) -> tensorflow.keras.Model:
        return self.decorated_ml_model.get_model()


    def train(self) -> tensorflow.keras.callbacks.History:
        return self.decorated_ml_model.train()

    
    def pretrain(self) -> tensorflow.keras.callbacks.History:
        return self.decorated_ml_model.pretrain()


    def inference(self) -> pandas.DataFrame:
        return self.decorated_ml_model.inference()