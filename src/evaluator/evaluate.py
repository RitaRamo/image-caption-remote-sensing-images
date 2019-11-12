import numpy as np
import tensorflow as tf
#from nlgeval import NLGEval
from nlgeval import compute_individual_metrics


class Evaluator():

    def __init__(self, generator, model, token_to_id, id_to_token):
        self.generator = generator
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    def evaluate(self, test_dataset):
        # nlgeval = NLGEval()  # loads the models
        for img_name, references_captions_of_image in test_dataset.items():

            img_tensor = self.generator.get_image(img_name)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            print("this is my image tensor", np.shape(img_tensor))

            text_generated = self.model.generate_text(
                img_tensor, self.token_to_id, self.id_to_token)

            print("how this is the caption", text_generated)
            scores = self.compare_results(
                references_captions_of_image, text_generated)
            break

    def compare_results(self, references_captions, predicted_caption):
        print("this are reds", references_captions)
        print("this are predicted_caption", predicted_caption)

        # metrics_dict = compute_individual_metrics(
        #     references_captions, predicted_caption)
        # print("metirc dict", metrics_dict)
