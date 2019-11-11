import numpy as np
import tensorflow as tf


class Evaluator():

    def __init__(self, generator, model, token_to_id, id_to_token):
        self.generator = generator
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    def evaluate(self, images_names, captions_of_tokens):

        dataset_size = len(images_names)

        for i in range(dataset_size):

            img_name = images_names[i]
            img_tensor = self.generator.get_image(img_name)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            print("this is my image tensor", np.shape(img_tensor))

            text_generated = self.model.generate_text(
                img_tensor, self.token_to_id, self.id_to_token)

            caption = captions_of_tokens[i]
            print("how this is the caption", caption)
            self.compare_results(text_generated, caption)
            break

    def compare_results(self, predicted_caption, caption):
        pass
