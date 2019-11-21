from abc import ABC, abstractmethod
import tensorflow as tf
import json

BATCH_SIZE = 2


class AbstractModel(ABC):

    EPOCHS = 1
    MODEL_DIRECTORY = "././experiments/results/"

    def __init__(
        self,
        model_name,
        vocab_size,
        max_len,
        token_to_id,
        id_to_token,
        encoder_input_size,
        embedding_type=None,
        embedding_size=300,
        lstm_units=256
    ):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.encoder_input_size = encoder_input_size
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.lstm_units = lstm_units
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
            metrics=['accuracy']
        )

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, verbose=1, restore_best_weights=True)

        self.model.fit_generator(
            train_dataset,
            epochs=self.EPOCHS,
            steps_per_epoch=1,  # len_train_dataset/BATCH_SIZE,
            validation_data=val_dataset,
            validation_steps=1,  # len_val_dataset/BATCH_SIZE,
            callbacks=[early_stop]
        )

    def get_path(self):
        return self.MODEL_DIRECTORY + 'trained_models/' + self.model_name+'.h5'

    def save(self):
        self.model.save(self.get_path())

    def load(self):
        self.model = tf.keras.models.load_model(self.get_path())

    def save_scores(self, scores):
        scores_path = self.MODEL_DIRECTORY + 'evaluation_scores/' + self.model_name
        with open(scores_path+'.json', 'w+') as f:
            json.dump(scores, f, indent=2)
