
import numpy as np
import tensorflow as tf
#from nlgeval import NLGEval
#from nlgeval import compute_metrics
from nlgeval import NLGEval
from collections import defaultdict
import logging


class EvaluatorIndividualMetrics():

    def __init__(self, generator, model):
        self.generator = generator
        self.model = model

    def evaluate(self, test_dataset, args):
        logging.info("start evaluating")

        predicted = {}

        metrics = {}
        if args.disable_metrics:
            logging.info(
                "disable_metrics = True, thus will not compute metrics")

        else:
            nlgeval = NLGEval()  # loads the models

        # ifi ==3 ver se dá
        n_comparations = 0

        for img_name, references in test_dataset.items():

            img_tensor = self.generator.get_image(img_name)
            #img_tensor = tf.expand_dims(img_tensor, axis=0)

            if args.beam_search:
                text_generated = self.model.beam_search(img_tensor)
            else:
                text_generated = self.model.generate_text(
                    img_tensor)

            if args.disable_metrics:
                break

            #logging.info("how this is the caption", text_generated)
            # scores = self.compare_results(
            #     references_captions_of_image, text_generated)

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

        logging.info("avg_metrics %s", avg_metrics)

        return predicted

    def compare_results(self, nlgeval, references_captions, predicted_captions):
        #logging.info("ref %s", references_captions)
        #logging.info("caption %s", predicted_captions)

        metrics_dict = nlgeval.compute_individual_metrics(
            references_captions, predicted_captions)
        logging.info("this are dic metrics %s", metrics_dict)
        return metrics_dict
