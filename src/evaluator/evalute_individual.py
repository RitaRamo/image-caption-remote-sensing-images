
import numpy as np
import tensorflow as tf
#from nlgeval import NLGEval
#from nlgeval import compute_metrics
from nlgeval import NLGEval
from collections import defaultdict


class EvaluatorIndividualMetrics():

    def __init__(self, generator, model):
        self.generator = generator
        self.model = model

    def evaluate(self, test_dataset):
        predicted = {}

        metrics = {}
        nlgeval = NLGEval()  # loads the models
        # ifi ==3 ver se dá
        n_comparations = 0

        for img_name, references in test_dataset.items():

            img_tensor = self.generator.get_image(img_name)
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            text_generated = self.model.generate_text(
                img_tensor)

            #print("how this is the caption", text_generated)
            # scores = self.compare_results(
            #     references_captions_of_image, text_generated)

            (text_generated)
            scores = self.compare_results(nlgeval, references, text_generated)

            predicted[img_name] = {
                "value": text_generated,
                "scores": scores
            }

            for metric, score in scores.items():
                if metric not in metrics:
                    metrics[metric] = score
                else:
                    metrics[metric] += score
            n_comparations += 1

        avg_metrics = {metric: total_score /
                       n_comparations for metric, total_score in metrics.items()}

        predicted['avg_metrics'] = {
            "value": "",
            "scores": avg_metrics
        }

        print("avg_metrics", avg_metrics)

        return predicted

    def compare_results(self, nlgeval, references_captions, predicted_captions):
        print("ref", references_captions)
        print("caption", predicted_captions)

        metrics_dict = nlgeval.compute_individual_metrics(
            references_captions, predicted_captions)
        print("this are dic metrics", metrics_dict)
        return metrics_dict
        # return {}

        # metrics_dict = compute_individual_metrics(
        #     references_captions, predicted_caption)
        # print("metirc dict", metrics_dict)
