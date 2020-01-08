from abc import ABC, abstractmethod
import tensorflow as tf
import json
import os
import logging
import shutil
import numpy as np


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
            logging.info(
                "Restore model from checkpoint. Start epoch %s ", start_epoch)

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

        # TODO: DAR A ultima loss do checkpoint (variable) à baseline :)
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

        # class EarlyStopping(tf.keras.callbacks.Callback):
        #     """Stop training when a monitored quantity has stopped improving.
        #     Arguments:
        #         monitor: Quantity to be monitored.
        #         min_delta: Minimum change in the monitored quantity
        #             to qualify as an improvement, i.e. an absolute
        #             change of less than min_delta, will count as no
        #             improvement.
        #         patience: Number of epochs with no improvement
        #             after which training will be stopped.
        #         verbose: verbosity mode.
        #         mode: One of `{"auto", "min", "max"}`. In `min` mode,
        #             training will stop when the quantity
        #             monitored has stopped decreasing; in `max`
        #             mode it will stop when the quantity
        #             monitored has stopped increasing; in `auto`
        #             mode, the direction is automatically inferred
        #             from the name of the monitored quantity.
        #         baseline: Baseline value for the monitored quantity.
        #             Training will stop if the model doesn't show improvement over the
        #             baseline.
        #         restore_best_weights: Whether to restore model weights from
        #             the epoch with the best value of the monitored quantity.
        #             If False, the model weights obtained at the last step of
        #             training are used.
        #     Example:
        #     ```python
        #     callback = tf.keras.callbacks.EarlyStopping(
        #         monitor='val_loss', patience=3)
        #     # This callback will stop the training when there is no improvement in
        #     # the validation loss for three consecutive epochs.
        #     model.fit(data, labels, epochs=100, callbacks=[callback],
        #         validation_data=(val_data, val_labels))
        #     ```
        #     """

        #     def __init__(self,
        #                  monitor='val_loss',
        #                  min_delta=0,
        #                  patience=0,
        #                  verbose=0,
        #                  mode='auto',
        #                  baseline=None,
        #                  restore_best_weights=False):
        #         super(EarlyStopping, self).__init__()

        #         self.monitor = monitor
        #         self.patience = patience
        #         self.verbose = verbose
        #         self.baseline = baseline
        #         self.min_delta = abs(min_delta)
        #         self.wait = 0
        #         self.stopped_epoch = 0
        #         self.restore_best_weights = restore_best_weights
        #         self.best_weights = None

        #         if mode not in ['auto', 'min', 'max']:
        #             logging.warning('EarlyStopping mode %s is unknown, '
        #                             'fallback to auto mode.', mode)
        #             mode = 'auto'

        #         if mode == 'min':
        #             print("sou np less")
        #             self.monitor_op = np.less
        #         elif mode == 'max':
        #             print("sou np great")

        #             self.monitor_op = np.greater
        #         else:
        #             print("co acc sou")
        #             if 'acc' in self.monitor:
        #                 print("np.greater")

        #                 self.monitor_op = np.greater
        #             else:
        #                 print("sou np less")

        #                 self.monitor_op = np.less

        #         print("self.monitor_op é msmo, ", self.monitor_op)

        #         if self.monitor_op == np.greater:
        #             self.min_delta *= 1
        #             print(" entrei no min delta*1")

        #         else:
        #             self.min_delta *= -1
        #             print("yeah entrei aqui no mindelta*-1")

        #         print("self.min_delta", self.min_delta)

        #     def on_train_begin(self, logs=None):
        #         # Allow instances to be re-used
        #         self.wait = 0
        #         self.stopped_epoch = 0
        #         if self.baseline is not None:
        #             self.best = self.baseline
        #         else:
        #             self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        #         print("this is self.best", self.best)

        #     def on_epoch_end(self, epoch, logs=None):
        #         current = self.get_monitor_value(logs)
        #         if current is None:
        #             return

        #         print("\ncurrent", current)
        #         print("self.min_delta", self.min_delta)
        #         print("self.best",  self.best)
        #         print("lets check having wait", self.wait)

        #         if self.monitor_op(current - self.min_delta, self.best):
        #             print("yeah entrou")

        #             self.best = current
        #             self.wait = 0
        #             if self.restore_best_weights:
        #                 self.best_weights = self.model.get_weights()
        #         else:
        #             print("saius")

        #             self.wait += 1
        #             print("increse wait", self.wait)
        #             if self.wait >= self.patience:
        #                 self.stopped_epoch = epoch
        #                 self.model.stop_training = True
        #                 if self.restore_best_weights:
        #                     if self.verbose > 0:
        #                         print(
        #                             'Restoring model weights from the end of the best epoch.')
        #                     self.model.set_weights(self.best_weights)

        #     def on_train_end(self, logs=None):
        #         if self.stopped_epoch > 0 and self.verbose > 0:
        #             print('Epoch %05d: early stopping' %
        #                   (self.stopped_epoch + 1))

        #     def get_monitor_value(self, logs):
        #         logs = logs or {}
        #         monitor_value = logs.get(self.monitor)
        #         if monitor_value is None:
        #             logging.warning('Early stopping conditioned on metric `%s` '
        #                             'which is not available. Available metrics are: %s',
        #                             self.monitor, ','.join(list(logs.keys())))
        #         return monitor_value

        # early_stop = EarlyStopping(min_delta=0.5, patience=3, verbose=1)

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
        # return self.MODEL_DIRECTORY + 'trained_models/' + str(self.args.__dict__)+'.h5'
        return self.MODEL_DIRECTORY + 'trained_models/' + self.args.file_name+'.h5'

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
            'evaluation_scores/' + \
            self.args.file_name  # str(self.args.__dict__)
        with open(scores_path+'.json', 'w+') as f:
            json.dump(scores, f, indent=2)

    @abstractmethod
    def generate_text(self):
        pass
