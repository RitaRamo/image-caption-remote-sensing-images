from abc import ABC, abstractmethod
import tensorflow as tf
import json
import os
import logging
import shutil


class AbstractModel(ABC):

    MODEL_DIRECTORY = "././experiments/results/"

    def __init__(
        self,
        args,
        vocab_size,
        max_len,
        token_to_id,
        id_to_token,
        encoder_input_size,
        embedding_type=None,
        embedding_size=300,
        units=256
    ):
        self.args = args
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.encoder_input_size = encoder_input_size
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.units = units
        self.model = None

    @abstractmethod
    def create(self):
        pass

    def summary(self):
        self.model.summary()

    def build(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # change to categorical accuracy!
        )

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):
        if self.args.disable_steps:
            train_steps = 1
            val_steps = 1
        else:
            train_steps = int(len_train_dataset/self.args.batch_size)
            val_steps = int(len_val_dataset/self.args.batch_size)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            verbose=1,
            restore_best_weights=True)

        ckpt = tf.train.Checkpoint(
            optimizer=self.model.optimizer, model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, './tf_ckpts', max_to_keep=2)

        start_epoch = 0
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            # codigo para fazer next do dataset o nº de bath size* steps*epochs
            # train_dataset.take(start_epoch*self.args.batch_size*train_steps)

        logging.info("start epoch is %s", start_epoch)

        class CheckCallback(tf.keras.callbacks.Callback):

            def __init__(self, ckpt_manager):
                super(CheckCallback, self).__init__()
                self.ckpt_manager = ckpt_manager

            def on_epoch_end(self, epoch, logs=None):
                self.ckpt_manager.save()

        check_callback = CheckCallback(ckpt_manager)

        self.model.fit(
            train_dataset,
            epochs=self.args.epochs,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            initial_epoch=start_epoch,
            callbacks=[early_stop, check_callback]
        )

    # def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):
    #     for value in train_dataset.take(1):
    #         print("train value", value)
    #     for value in train_dataset.take(1):
    #         print("train value", value)

    #     if self.args.disable_steps:
    #         train_steps = 1
    #         val_steps = 1
    #     else:
    #         train_steps = int(len_train_dataset/self.args.batch_size)
    #         val_steps = int(len_val_dataset/self.args.batch_size)

    #     # Create a callback that saves the model's weights
    #     # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix,
    #     #                                                  save_weights_only=False,
    #     #                                                  verbose=1)

    #     # TODO: check this
    #     class CheckCallback(tf.keras.callbacks.Callback):

    #         def __init__(self, model):
    #             super(CheckCallback, self).__init__()
    #             self.basic_model = model

    #         def on_epoch_end(self, epoch, logs=None):
    #             self.basic_model.save()
    #             print("thi is my epoch", epoch)
    #             with open(self.basic_model.get_path()+'epoch.txt', 'w') as the_file:
    #                 the_file.write(str(epoch+1))

    #     check_callback = CheckCallback(self)

    #     start_epoch = 0

    #     if os.path.isfile(self.get_path()):
    #         self.load()
    #         with open(self.get_path()+'epoch.txt') as f:
    #             epoch = f.readline(1)
    #             print("this is my epoch to load", epoch)
    #             start_epoch = int(epoch)
    #     else:
    #         print("file does not exist")

    #     print("this is start_epoch", start_epoch)

    #     early_stop = tf.keras.callbacks.EarlyStopping(
    #         monitor='loss',
    #         patience=3,
    #         verbose=1,
    #         restore_best_weights=True)

    #     # latest = tf.train.latest_checkpoint(self.checkpoint_dir)
    #     # if latest is not None:
    #     #     self.model.load_weights(latest)
    #     #     start_epoch = int(latest.split('_')[-1])
    #     # else:
    #     #     start_epoch = 0

    #     print("this is the start epoch", start_epoch)

    #     self.model.fit_generator(
    #         train_dataset,
    #         epochs=3,  # self.args.epochs,
    #         steps_per_epoch=train_steps,
    #         validation_data=val_dataset,
    #         validation_steps=val_steps,
    #         initial_epoch=start_epoch,
    #         callbacks=[early_stop, check_callback]
    #     )

    def get_path(self):
        return self.MODEL_DIRECTORY + 'trained_models/' + str(self.args.__dict__)+'.h5'

    def save(self):
        try:
            self.model.save(self.get_path())
            logging.info("model saved")
        except:
            logging.warning("saving model did not succeed")
        else:
            if os.path.exists("tf_ckpts"):
                shutil.rmtree("tf_ckpts")
                # os.rmdir("tf_ckpts")
                print("model saved, thus removing checkpoints")
            else:
                print("unable to remove checkpoints, does not exist")

    def load(self):
        self.model = tf.keras.models.load_model(self.get_path())

    def save_scores(self, scores):
        scores = {key: str(values) for key, values in scores.items()}

        scores_path = self.MODEL_DIRECTORY + \
            'evaluation_scores/' + str(self.args.__dict__)
        with open(scores_path+'.json', 'w+') as f:
            json.dump(scores, f, indent=2)
