import numpy as np
import tensorflow as tf
import logging


class EarlyStoppingWithCheckpoint():

    def __init__(self,  ckpt, ckpt_manager, baseline, min_delta=0, patience=0):
        super(EarlyStoppingWithCheckpoint, self).__init__()

        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager
        self.baseline = baseline
        self.min_delta = abs(min_delta) * -1
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf

    def on_epoch_end(self, epoch, current):
        if current is None:
            return

        if np.less(current - self.min_delta, self.best):

            self.best = current
            self.wait = 0
            self.ckpt.loss.assign(self.best)
            self.ckpt_manager.save()

        else:

            self.wait += 1
            logging.info("Val without improvement. Not Saving")

            if self.wait >= self.patience:
                logging.info("Early stopping")
                self.stopped_epoch = epoch
                self.stop_training = True

    def is_to_stop_training(self):
        return self.stop_training
