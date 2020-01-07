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
        self.checkpoint_path = None

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

    @abstractmethod
    def _checkpoint(self):
        pass

    def _load_latest_checkpoint(self):
        ckpt = self._checkpoint()
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.checkpoint_path, max_to_keep=2)  # TODO:OPAAA PATH!!...

        start_epoch = 0
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            logging.info("Restore model from checkpoint")

        return ckpt_manager, start_epoch

    def _get_steps(self, len_train_dataset, len_val_dataset):
        if self.args.disable_steps:  # pode sair para o abstract model get_steps
            train_steps = 1
            val_steps = 1
        else:
            train_steps = int(len_train_dataset/self.args.batch_size)
            val_steps = int(len_val_dataset/self.args.batch_size)

        return train_steps, val_steps

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):
        train_steps, val_steps = self._get_steps(
            len_train_dataset, len_val_dataset)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            verbose=1,
            restore_best_weights=True)

        ckpt_manager, start_epoch = self._load_latest_checkpoint()
        logging.info("start epoch is %s", start_epoch)

        class CheckCallback(tf.keras.callbacks.Callback):

            def __init__(self, ckpt_manager):
                super(CheckCallback, self).__init__()
                self.ckpt_manager = ckpt_manager

            def on_epoch_end(self, epoch, logs=None):
                self.ckpt_manager.save()

        check_callback = CheckCallback(ckpt_manager)

        self.model.fit_generator(
            train_dataset,
            epochs=self.args.epochs,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            initial_epoch=start_epoch,
            callbacks=[early_stop, check_callback]
        )

    def get_path(self):
        return self.MODEL_DIRECTORY + 'trained_models/' + str(self.args.__dict__)+'.h5'

    def save(self):
        try:
            self.model.save(self.get_path())
            logging.info("model saved")
        except Exception as e:
            logging.warning("saving model did not succeed %s", e)
        else:
            if os.path.exists(self.checkpoint_path):  # "tf_ckpts"):
                shutil.rmtree(self.checkpoint_path)  # "tf_ckpts")
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

    @abstractmethod
    def generate_text(self):
        pass
