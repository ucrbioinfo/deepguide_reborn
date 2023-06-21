import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disables Tensorflow logging about CPU instructions.
import sys
import pandas
import argparse
import tensorflow

from abc import ABC, abstractmethod


class MLModel(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        self.pretraining_data: dict = dict()
        self.training_data: dict = dict()
        self.inference_data: dict = dict()
        self.callbacks: list[tensorflow.keras.callbacks.Callback] = list()

        self.args = args

        self.exp_directory = os.path.join(self.args.output_directory, self.args.experiment_name, '')
        self.exp_model_directory = os.path.join(self.exp_directory, 'model', '')
        self.exp_weights_directory = os.path.join(self.exp_directory, 'weights', '')

        self.pretrain_model_path = os.path.join(self.exp_model_directory, self.args.cas + '_' + self.args.model + '_' + str(self.args.guide_length) + 'nt_pretrain_model.h5')
        self.pretrain_weights_path = os.path.join(self.exp_weights_directory, self.args.cas + '_' + self.args.model + '_' + str(self.args.guide_length) + 'nt_pretrain_weights.h5')
        self.train_model_path = os.path.join(self.exp_model_directory, self.args.cas + '_' + self.args.model + '_' + str(self.args.guide_length) + 'nt_train_model.h5')
        self.train_weights_path = os.path.join(self.exp_weights_directory, self.args.cas + '_' + self.args.model + '_' + str(self.args.guide_length) + 'nt_train_weights.h5')

        lib_name = self.args.inference_guides_csv_file_name.split('.csv')[0]
        self.inference_output_path = os.path.join(self.exp_directory, lib_name + '_' + self.args.cas + '_' + self.args.model + '_' + str(self.args.guide_length) + 'nt_predicted_scores.csv')

        mode = self.args.mode

        if (mode == 'pretrain' or mode == 'pt' or mode == 'pti'):
            if os.path.exists(self.pretrain_model_path):
                print('Error: Pretrained model already exists: {path}'.format(path=self.pretrain_model_path))
                print('Please rename the experiment or make a backup.')
                sys.exit(1)
            if os.path.exists(self.pretrain_weights_path):
                print('Error: Pretrain weights already exist: {path}'.format(path=self.pretrain_weights_path))
                print('Please rename the experiment or make a backup.')
                sys.exit(1)

        if (mode == 'train' or mode == 'pt' or mode == 'pti' or mode == 'ti'):
            if os.path.exists(self.train_model_path):
                print('Error: Trained model already exists: {path}'.format(path=self.train_model_path))
                print('Please rename the experiment or make a backup.')
                sys.exit(1)
            if os.path.exists(self.train_weights_path):
                print('Error: Trained weights already exist: {path}'.format(path=self.train_weights_path))
                print('Please rename the experiment or make a backup.')
                sys.exit(1)


        # if (mode == 'pti' or mode == 'ti') and os.path.exists(self.inference_output_path):
        #     print('Error: Inference output file already exists: {path}'.format(self.inference_output_path))
        #     print('Please rename the experiment or make a backup.')
        #     sys.exit(1)

        try:
            os.makedirs(self.exp_model_directory)
            os.makedirs(self.exp_weights_directory)
        except FileExistsError:
            pass
        

    @abstractmethod
    def get_model(self) -> tensorflow.keras.Model:
        '''
        Create and return a tensorflow.keras.Model model.
        '''


    @abstractmethod
    def pretrain(self) -> tensorflow.keras.callbacks.History:
        '''
        Pretrain a model and return a History object.
        '''


    @abstractmethod
    def train(self) -> tensorflow.keras.callbacks.History:
        '''
        Train a model and return a history object.
        '''


    @abstractmethod
    def inference(self) -> pandas.DataFrame:
        '''
        Perform inference on an input file (configured in config.yaml)
        and return a pandas DataFrame.
        '''