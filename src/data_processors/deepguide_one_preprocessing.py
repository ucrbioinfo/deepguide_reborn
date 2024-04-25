import os
import sys
import numpy
import pandas
import argparse
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from data_processors.preprocessing_base import PreprocessingBase


class DeepGuideOnePreprocessing(PreprocessingBase):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    
    def cut_genome(self, output_csv: bool = True) -> pandas.DataFrame:
        guide_length = self.args.guide_length
        input_genome_path = os.path.join(self.args.input_directory, self.args.pretrain_genome_input_name)
        output_kmers_path = os.path.join(self.args.output_directory, self.args.experiment_name, 'kmers_from_genome.csv')

        try:
            print('Looking for genome in {path}'.format(path=input_genome_path))
            records = list(SeqIO.parse(input_genome_path, 'fasta'))
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

    
        print('genome_cutter.py found {n} chromosomes/records.'.format(n=len(records)))
        print('Generating {path}.'.format(path=output_kmers_path))

        sequence = str()
        for record in records:
            sequence += str(record.seq).upper()

        kmers = list()
        for i in range(0, len(sequence)-guide_length):
            kmers.append(sequence[i:i+guide_length])

        kmers_df = pandas.DataFrame({'kmers': kmers})

        if output_csv:
            kmers_df.to_csv(
                output_kmers_path,
                index=False,
            )

            print('Genome cutter saved {num} kmers for pretraining in {path}.'.format(
                num=len(kmers_df), 
                path=output_kmers_path,
                )
            )

        return kmers_df


    def encode_guides(self, guides: list[str]) -> numpy.ndarray:
        data_length = len(guides)
        tensor = numpy.zeros((data_length, self.args.guide_length, 4), dtype=int)

        for i in range(data_length):
            guide = guides[i]
            for j in range(self.args.guide_length):
                if   guide[j] in 'Aa': tensor[i, j, 0] = 1
                elif guide[j] in 'Cc': tensor[i, j, 1] = 1
                elif guide[j] in 'Gg': tensor[i, j, 2] = 1
                elif guide[j] in 'Tt': tensor[i, j, 3] = 1

        return tensor


    def preprocess_pretrain(self) -> dict:
        print('Cutting the genome for pretraining.')
        kmers_df = self.cut_genome()
        print('Done.')

        print('Loading and shuffling pretrain data.')
        onehot_encoded = self.encode_guides(kmers_df['kmers'].sample(frac=1).to_list())
        print('Done.')

        train_x, valid_x = train_test_split(
            onehot_encoded,
            train_size=self.args.dg_one_pretrain_train_test_ratio,
        )

        print('Pretraining DeepGuide 1 on {train_n} and validating on {valid_n} examples. Ratio: {ratio}.'.format(
            train_n=len(train_x),
            valid_n=len(valid_x),
            ratio=self.args.dg_one_pretrain_train_test_ratio,
        ))

        return dict({
            'train_x': train_x,
            'valid_x': valid_x,
        })


    def preprocess_train(self) -> dict:
        input_path = os.path.join(self.args.input_directory, self.args.train_guides_csv_file_name)
        
        print(f'Loading training data from {input_path}')

        train_df = None
        try:
            train_df = pandas.read_csv(input_path).sample(frac=1)  # Shuffles the dataset.
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
            
        print('Done.')

        X = None
        try:
            X = train_df[self.args.train_guide_seq_col_name].to_list()
        except KeyError as e:
            print(e)
            print(f'No such column as {self.args.train_guide_seq_col_name} in train csv file. Did you misspell it?')
            sys.exit(1)

        Y = None
        try:
            Y = train_df[self.args.train_guide_score_col_name].to_list()
        except KeyError as e:
            print(e)
            print(f'No such column as {self.args.train_guide_score_col_name} in train csv file. Did you misspell it?')
            sys.exit(1)

        nu_list = [1] * len(X)
        if 'nucleosome' in self.args.cas:
            try:
                nu_list = train_df[self.args.train_nucleosome_occupancy_col_name].tolist()
                print(f'Loaded nucleosome occupancy data from {self.args.train_nucleosome_occupancy_col_name}.')
            except KeyError as e:
                print(f'Training nucleosome occupancy column {self.args.train_nucleosome_occupancy_col_name} not found. Exiting.')
                sys.exit(1)

        X = self.encode_guides(X)
        Y = numpy.asarray(Y).reshape((-1, 1))  # column vector
        NU = numpy.array(nu_list)

        (train_x, valid_x,
         train_nu, valid_nu,
         train_y, valid_y) = train_test_split(X,
                                               NU, 
                                               Y, 
                                               train_size=self.args.dg_one_train_test_ratio)

        print('Training DeepGuide 1 on {train_n} and validating on {valid_n} examples. Ratio: {ratio}.'.format(
            train_n=len(train_x),
            valid_n=len(valid_x),
            ratio=self.args.dg_one_train_test_ratio,
        ))

        return dict({
            'train_x': train_x,
            'train_nu': train_nu,
            'train_y': train_y,
            'valid_x': valid_x,
            'valid_nu': valid_nu,
            'valid_y': valid_y,
        })


    def preprocess_inference(self) -> dict:
        path = os.path.join(self.args.input_directory, self.args.inference_guides_csv_file_name)

        test_df = None
        try:
            test_df = pandas.read_csv(path)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        try:
            test_x_raw = test_df[self.args.inference_guide_seq_col_name].to_list()
        except KeyError as e:
            print(e)
            print(f'No such column as {self.args.inference_guide_seq_col_name} in inference csv file. Did you misspell it?')
            sys.exit(1)
        
        nu_list = list()
        if 'nucleosome' in self.args.cas:
            try:
                nu_list = test_df[self.args.inference_nucleosome_occupancy_col_name]
                print(f'Loaded nucleosome occupancy data from {self.args.inference_nucleosome_occupancy_col_name}.')
            except KeyError as e:
                print(f'No such column as {self.args.inference_nucleosome_occupancy_col_name} in inference csv file. Did you misspell it?. Exiting.')
                sys.exit(1)
        
        test_x = self.encode_guides(test_x_raw)

        return dict({
            'test_x': test_x,
            'test_nu': nu_list,
            'guides': test_x_raw,
        })
    