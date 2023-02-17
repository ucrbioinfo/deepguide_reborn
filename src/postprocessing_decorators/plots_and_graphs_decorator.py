import os
import pandas
import argparse
import tensorflow
import matplotlib.pyplot
import sklearn.metrics
import sklearn.preprocessing

from ml_models.ml_model_base import MLModel
from postprocessing_decorators.postprocessing_decorator_base import PostprocessingDecoratorBase


class ROCCurve(tensorflow.keras.callbacks.Callback):
    def __init__(self, valid_x, valid_y, threshold: float, output_directory: str) -> None:
        super().__init__()

        self.valid_x = valid_x
        self.valid_y = valid_y
        self.threshold = threshold
        self.output_directory = output_directory
        self.metrics_output_path = os.path.join(
            output_directory,
            'metrics.txt',
        )


    def on_train_end(self, logs=None) -> None:
        pred_y = self.model.predict(self.valid_x)
        pred_y = sklearn.preprocessing.binarize(X=pred_y, threshold=self.threshold)
        valid_y = sklearn.preprocessing.binarize(X=self.valid_y, threshold=self.threshold)

        auc = sklearn.metrics.roc_auc_score(valid_y, pred_y)
        fpr, tpr, _ = sklearn.metrics.roc_curve(valid_y, pred_y)

        label = 'AUC = {auc:0.3f}'.format(auc=auc)
        matplotlib.pyplot.plot(fpr, tpr, label=label)
        matplotlib.pyplot.ylabel('True Positive Rate')
        matplotlib.pyplot.xlabel('False Positive Rate')
        matplotlib.pyplot.legend(loc='lower right')
        matplotlib.pyplot.savefig(self.output_directory + 'validation_roc_curve.png')

        with open(self.metrics_output_path, 'a') as f:
            output = 'AUC-ROC for validation split ({n} examples): {auc:0.3f}\n'.format(
                n=len(pred_y),
                auc=auc,
            )
            f.write(output)


class PlotsAndGraphsDecorator(PostprocessingDecoratorBase):
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
        
        roc_curve = ROCCurve(
            valid_x=self.decorated_ml_model.training_data['valid_x'],
            valid_y=self.decorated_ml_model.training_data['valid_y'],
            threshold=self.args.dg_one_roc_curve_threshold,
            output_directory=output_directory,
            )

        self.decorated_ml_model.callbacks.append(roc_curve)

        return self.decorated_ml_model.train()


    def inference(self) -> pandas.DataFrame:
        return self.decorated_ml_model.inference()
