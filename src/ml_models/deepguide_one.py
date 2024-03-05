import pandas
import argparse
import tensorflow

from ml_models.ml_model_base import MLModel


class DeepGuideOne(MLModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        callback_list = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1,
            ),
            tensorflow.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=0.001
            ),
        ]

        self.callbacks.extend(callback_list)


    def get_model(self) -> tensorflow.keras.Model:
        ''' 
        Model developed by Dipankar Ranjan Baisya https://github.com/dDipankar/DeepGuide
        Presented in paper https://rdcu.be/cUM5w 
        Genome-wide functional screens enable the prediction of 
        high activity CRISPR-Cas9 and -Cas12a guides in Yarrowia lipolytica.
        Nat Commun 13, 922 (2022). https://doi.org/10.1038/s41467-022-28540-0
        Dipankar Baisya, Adithya Ramesh, Cory Schwartz, Stefano Lonardi & Ian Wheeldon 
        '''

        encoder_input = tensorflow.keras.Input(shape=(self.args.guide_length, 4), name='encoder_input')
        encoder = tensorflow.keras.layers.Conv1D(activation='relu', padding='same', strides=1, filters=20, kernel_size=5, kernel_initializer='glorot_uniform', kernel_regularizer=tensorflow.keras.regularizers.l2(0.0001))(encoder_input)
        encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
        encoder = tensorflow.keras.layers.MaxPooling1D(pool_size=(2))(encoder)
        encoder = tensorflow.keras.layers.Conv1D(activation='relu', padding='same', strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform', kernel_regularizer=tensorflow.keras.regularizers.l2(0.0001))(encoder)
        encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
        encoder_output = tensorflow.keras.layers.AveragePooling1D(pool_size=(2), name='encoder_output')(encoder)

        decoder = tensorflow.keras.layers.UpSampling1D(size=2)(encoder_output)
        decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
        decoder = tensorflow.keras.layers.Conv1D(activation='relu', padding='same', strides=1, filters=40, kernel_size=8, kernel_initializer='glorot_uniform', kernel_regularizer=tensorflow.keras.regularizers.l2(0.0001))(decoder)
        decoder = tensorflow.keras.layers.UpSampling1D(size=2)(decoder)
        decoder_output = tensorflow.keras.layers.Conv1D(activation='relu', padding='same', strides=1, filters=4, kernel_size=5, kernel_initializer='glorot_uniform', kernel_regularizer=tensorflow.keras.regularizers.l2(0.0001), name='decoder_output')(decoder)
        auto_encoder = tensorflow.keras.Model(inputs=encoder_input, outputs=decoder_output)

        return auto_encoder


    def pretrain(self) -> tensorflow.keras.callbacks.History:
        print('Pretraining DeepGuide 1...')
        data_dict = self.pretraining_data
        assert(len(data_dict) > 0)

        auto_encoder = self.get_model()
        auto_encoder.summary()

        if self.args.plot_model:
            tensorflow.keras.utils.plot_model(
                model=auto_encoder, 
                to_file=(self.exp_directory + self.args.cas + '_dg1_pretraining_model.png'),
                show_shapes=True,
            )

        auto_encoder.compile(
            loss=tensorflow.keras.losses.BinaryCrossentropy(),
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.args.dg_one_adam_lr)
        )

        print(f'Pretraining auto-encoder with batch size {self.args.dg_one_pretrain_batch_size}.')
        print(f'Pretraining auto-encoder with {self.args.dg_one_pretrain_epochs} epochs.')

        history = auto_encoder.fit(
            x=data_dict['train_x'],
            y=data_dict['train_x'],
            batch_size=self.args.dg_one_pretrain_batch_size,
            epochs=self.args.dg_one_pretrain_epochs,
            shuffle=True,
            validation_data=(data_dict['valid_x'], data_dict['valid_x']),
        )

        print('Saving pretrained model to {path}'.format(path=self.pretrain_model_path))
        auto_encoder.save(self.pretrain_model_path)
        print('Saving pretrained weights to {path}'.format(path=self.pretrain_weights_path))
        auto_encoder.save_weights(self.pretrain_weights_path)
        
        print('Done pretraining DeepGuide 1.')
        return history


    def train(self) -> tensorflow.keras.callbacks.History:
        print('Training DeepGuide 1...')
        data_dict = self.training_data
        assert(len(data_dict) > 0)

        auto_encoder = self.get_model()

        # Do we need to load pretrained weights? Depends on the selected mode in config.yaml
        if self.args.mode == 'pretrain' or self.args.mode == 'pt' or self.args.mode == 'pti':
            print('Training with pretrained weights {path}'.format(path=self.pretrain_weights_path))
            auto_encoder.load_weights(self.pretrain_weights_path)

            for layer in auto_encoder.layers:
                layer.trainable = True

        # Disregard the decoder - Continue training using the (pretrained) encoder.
        flatten = tensorflow.keras.layers.Flatten()(auto_encoder.get_layer('encoder_output').output)
        fc = tensorflow.keras.layers.Dropout(rate=0.5)(flatten)
        fc = tensorflow.keras.layers.Dense(units=80, activation='relu', kernel_initializer='glorot_uniform')(fc)
        fc = tensorflow.keras.layers.Dropout(rate=0.5)(fc)
        output = tensorflow.keras.layers.Dense(units=1, activation='linear')(fc) 
        model = tensorflow.keras.Model(inputs=auto_encoder.input, outputs=output)
        
        model.summary()
        
        if self.args.plot_model:
            tensorflow.keras.utils.plot_model(
                model=model, 
                to_file=self.exp_directory + self.args.cas + '_dg1_training_model.png',
                show_shapes=True,
            )

        model.compile(
            loss=tensorflow.keras.losses.MeanSquaredError(),
            optimizer=tensorflow.keras.optimizers.Adam(self.args.dg_one_adam_lr),
        )

        history = model.fit(
            x=data_dict['train_x'],
            y=data_dict['train_y'],
            batch_size=self.args.dg_one_batch_size,
            epochs=self.args.dg_one_epochs,
            shuffle=True,
            validation_data=(data_dict['valid_x'], data_dict['valid_y']),
            callbacks=self.callbacks,
        )

        print('Saving trained model to {path}'.format(path=self.train_model_path))
        model.save(self.train_model_path)
        print('Saving trained weights to {path}'.format(path=self.train_weights_path))
        model.save_weights(self.train_weights_path)

        print('Done training DeepGuide 1.')
        return history


    def inference(self) -> pandas.DataFrame:
        data_dict = self.inference_data
        assert(len(data_dict) > 0)

        model = tensorflow.keras.models.load_model(self.train_model_path)

        model.summary()
        
        if self.args.plot_model:
            tensorflow.keras.utils.plot_model(
                model=model, 
                to_file=self.exp_directory + self.args.cas + '_dg1_inference_model.png',
                show_shapes=True,
            )

        print('Inference with trained weights {path}'.format(path=self.train_weights_path))
        model.load_weights(self.train_weights_path)

        pred_y = model.predict(data_dict['test_x']).flatten()

        output_dataframe = pandas.DataFrame({
            'guide_with_context': data_dict['guides'],
            'predicted_dg1_score': pred_y,
        })

        output_dataframe.to_csv(self.inference_output_path, index=False)
        print('Saved predictions to {path}'.format(path=self.inference_output_path))
        
        return output_dataframe
