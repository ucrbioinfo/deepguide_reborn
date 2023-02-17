import numpy
import argparse
from abc import ABC, abstractmethod


class PreprocessingBase:
    def __init__(self, args: argparse.Namespace) -> None:
        pass


    @abstractmethod
    def encode_guides(self, guides: list[str]) -> numpy.ndarray:
        '''
        ## Args:
            * guides: a list of fixed-size k-mers (aka guide RNAs)
            These must consist of only the 4 letter alphabet ACTG.

        ## Returns:
            A 3d-tensor containing onehot encoded rows for each sequence in `guides`.
            E.g., [
                [ [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0] ], 
                [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]
            ]
            for ['TGCA', 'ACGT'].
        '''


    @abstractmethod
    def preprocess_pretrain(self) -> dict:
        '''
        Currently used only for DeepGuide 1. Cuts the genome in guide_length
        kmers (configured in config.yaml) and encodes them.
        '''


    @abstractmethod
    def preprocess_train(self) -> dict:
        '''
        Loads the input file. Shuffles the dataset.
        Onehot encodes, and splits into train/test. 
        
        ## Returns:
        dict({
            'train_x': train_x,
            'train_y': train_y,
            'valid_x': valid_x,
            'valid_y': valid_y,
        })

        Where values are returned by sklearn.model_selection.train_test_split
        '''


    @abstractmethod
    def preprocess_inference(self) -> dict:
        '''
        Reads the inference input file (inference_guides_csv_file_name in config.yaml).
        Encodes the input sequence column and returns as a dict:
        
        return dict({
            'guides': test_x_raw,  # Non-onehot encoded sequences
            'test_x': test_x,
        })
        '''
