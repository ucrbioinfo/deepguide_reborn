import os
import pandas
import argparse
import tensorflow
import scipy.stats

from ml_models.ml_model_base import MLModel
from postprocessing_decorators.postprocessing_decorator_base import PostprocessingDecoratorBase


class PearsonrCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, output_directory: str, valid_x, valid_y) -> None:
        super().__init__()

        self.valid_x = valid_x
        self.valid_y = valid_y
        self.output_directory = output_directory
        self.metrics_output_path = os.path.join(
            output_directory,
            'metrics.txt',
        )


    def on_train_end(self, logs=None) -> None:
        pred_y = self.model.predict(self.valid_x)

        print(self.valid_y)
        print(pred_y)

        try:
            score = scipy.stats.pearsonr(self.valid_y.flatten(), pred_y.flatten())[0]
        except ValueError as e:
            print(e)
            return
        
        with open(self.metrics_output_path, 'a') as f:
            output = 'Pearson r for validation split ({n} examples): {score:0.3f}\n'.format(
                n=len(pred_y),
                score=score,
            )
            f.write(output)


class SpearmanrCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, output_directory: str, valid_x, valid_y) -> None:
        super().__init__()

        self.valid_x = valid_x
        self.valid_y = valid_y
        self.output_directory = output_directory
        self.metrics_output_path = os.path.join(
            output_directory,
            'metrics.txt',
        )


    def on_train_end(self, logs=None) -> None:
        pred_y = self.model.predict(self.valid_x)

        try:
            score = scipy.stats.spearmanr(self.valid_y.flatten(), pred_y.flatten())[0]
        except ValueError as e:
            print(e)
            return
        
        with open(self.metrics_output_path, 'a') as f:
            output = 'Spearman r for validation split ({n} examples): {score:0.3f}\n'.format(
                n=len(pred_y),
                score=score,
            )
            f.write(output)


class MetricsDecorator(PostprocessingDecoratorBase):
    def __init__(self, decorated_ml_model: MLModel, args: argparse.Namespace) -> None:
        super().__init__(decorated_ml_model=decorated_ml_model, args=args)


    # Dynamic getters and setters
    # Delegates the getting and setting of attributes to decorated_ml_model
    def __getattr__(self, name):
        return getattr(self.__dict__['decorated_ml_model'], name)
    

    def __setattr__(self, name, value):
        if name in ('decorated_ml_model', 'args'):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['decorated_ml_model'], name, value)

    
    def __delattr__(self, name):
        delattr(self.__dict__['decorated_ml_model'], name)

    
    def get_model(self) -> tensorflow.keras.Model:
        return self.decorated_ml_model.get_model()


    def pretrain(self) -> tensorflow.keras.callbacks.History:
        return self.decorated_ml_model.pretrain()


    def train(self) -> tensorflow.keras.callbacks.History:
        output_directory = os.path.join(
            self.args.output_directory,
            self.args.experiment_name,
            '',
        )
    
        if 'nucleosome' in self.args.cas:
            valid_x = [self.decorated_ml_model.training_data['valid_x'], 
                       self.decorated_ml_model.training_data['valid_nu']]
        else:
            valid_x = self.decorated_ml_model.training_data['valid_x']

        valid_y = self.decorated_ml_model.training_data['valid_y']
        
        pearsonr_callback = PearsonrCallback(
            output_directory=output_directory,
            valid_x=valid_x,
            valid_y=valid_y
        )

        spearmanr_callback = SpearmanrCallback(
            output_directory=output_directory,
            valid_x=valid_x,
            valid_y=valid_y,
        )

        self.decorated_ml_model.callbacks.append(pearsonr_callback)
        self.decorated_ml_model.callbacks.append(spearmanr_callback)

        print('Find metrics in {path}'.format(path=output_directory + 'metrics.txt'))
        
        return self.decorated_ml_model.train()


    def inference(self) -> pandas.DataFrame:
        return self.decorated_ml_model.inference()
