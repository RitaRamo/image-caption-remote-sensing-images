import numpy as np
import tensorflow as tf
#from nlgeval import NLGEval
#from nlgeval import compute_metrics
from nlgeval import NLGEval


class Evaluator():

    def __init__(self, generator, model):
        self.generator = generator
        self.model = model

    def evaluate(self, test_dataset):
        predicted = []
        # nlgeval = NLGEval()  # loads the models
        # ifi ==3 ver se dá
        i = 0
        for img_name in test_dataset.keys():

            img_tensor = self.generator.get_image(img_name)
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            text_generated = self.model.generate_text(
                img_tensor)

            #print("how this is the caption", text_generated)
            # scores = self.compare_results(
            #     references_captions_of_image, text_generated)
            predicted.append(text_generated)
            i += 1
            if i == 5:
                break

        references_captions = list(test_dataset.values())[:i]

        print("len ref", len(references_captions))
        print("len caption", len(predicted))
        print("equal ", len(references_captions) == len(predicted))

        scores = self.compare_results(
            references_captions, predicted)
        # so funca se tivr + q um elemento, senão usar o outro para comparar!
        # TODO: fazer score individualmnt, guardando as categorias do maior score e as do pior

        return scores

    def compare_results(self, references_captions, predicted_captions):
        print("this are reds", references_captions)
        print("this are predicted_caption", predicted_captions)

        nlgeval = NLGEval()  # loads the models
        metrics_dict = nlgeval.compute_metrics(
            references_captions, predicted_captions)
        print("this are dic metrics", metrics_dict)
        return metrics_dict
        # return {}

        # metrics_dict = compute_individual_metrics(
        #     references_captions, predicted_caption)
        # print("metirc dict", metrics_dict)
