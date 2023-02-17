from data_processors.preprocessing_base import PreprocessingBase


class DeepGuideTwoPreprocessing(PreprocessingBase):
    def __init__(self, args):
        self.args = args


    def encode_guides(self, guides: list[str]):
        raise NotImplementedError


    def preprocess_train(self):
        raise NotImplementedError


    def preprocess_pretrain(self):
        raise NotImplementedError


    def preprocess_inference(self):
        raise NotImplementedError
        