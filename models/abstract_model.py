from abc import ABC, abstractmethod
import tensorflow as tf

BATCH_SIZE = 1


class AbstractModel(ABC):

    EPOCHS = 30
    MODEL_DIRECTORY = "./trained_models/"

    def __init__(self, vocab_size, max_len, encoder_input_size=131072, lstm_units=256, embedding_size=300):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.encoder_input_size = encoder_input_size
        self.lstm_units = lstm_units
        self.embedding_size = embedding_size
        self.model = None
        self.model_name = None

    @abstractmethod
    def create(self):
        pass

    def build(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, verbose=1, restore_best_weights=True)

        self.model.fit_generator(train_dataset, epochs=self.EPOCHS, steps_per_epoch=(
            BATCH_SIZE),
            # len_train_dataset/BATCH_SIZE),
            validation_data=val_dataset,
            # validation_steps=len_val_dataset/BATCH_SIZE,
            validation_steps=BATCH_SIZE,
            callbacks=[early_stop]
        )

    def get_path(self):
        return self.MODEL_DIRECTORY + self.model_name+'.h5'

    def save(self):
        self.model.save(self.get_path())

    def load(self):
        self.model = tf.keras.models.load_model(self.get_path())
