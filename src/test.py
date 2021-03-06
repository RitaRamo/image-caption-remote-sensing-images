import logging
from args_parser import get_args
import pandas as pd
from dataset import get_dataset, get_vocab_info
from evaluator.evaluate import Evaluator
from evaluator.evalute_individual import EvaluatorIndividualMetrics
from generators.abstract_generator import PATH
from generators.features_extracted.simple_generator import \
    FeaturesExtractedSimpleGenerator
from generators.fine_tuned.simple_generator import FineTunedSimpleGenerator
from models.simple_model import SimpleModel
from models.basic_model import BasicModel

from models.simple_model_finetuning import SimpleFineTunedModel
from models.attention_model import AttentionModel
from models.attention_model_with_continuos import AttentionContinuosModel
from models.attention_model_with_enc_initial_state import AttentionEncInitialStateModel
from models.attention_model_lstm import AttentionLSTMModel
from models.attention_model_lstm_dropout import AttentionLSTMDroupoutModel
from models.attention_model_lstm_regularizer import AttentionLSTMRegularizerModel
from models.attention_model_without_mask import AttentionLSTMWithoutMaskModel

# from models.attention_model_with_all_context import AttentionAllContextModel
# from models.attention_model_with_all_context2 import AttentionAllContextModel2

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    args = get_args()
    logging.info(args.__dict__)

    raw_dataset = pd.read_json(PATH + "dataset_rsicd.json")

    vocab_info = get_vocab_info("src/datasets/RSICD/dataset2/")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    test_dataset = get_dataset(
        "src/datasets/RSICD/dataset2/test.json")

    generator_args = (raw_dataset, args.image_model_type)

    if args.fine_tuning:
        logging.info("fine tuning")
        generator = FineTunedSimpleGenerator(*generator_args)

    else:
        logging.info("feature extraction")

        generator = FeaturesExtractedSimpleGenerator(*generator_args)

    model_class = globals()[args.model_class_str]

    model = model_class(
        args,
        vocab_size,
        max_len,
        token_to_id,
        id_to_token,
        generator.get_shape_of_input_image(),
        args.embedding_type,
        args.units
    )

    logging.info("loading the model")

    model.load()

    evaluator = EvaluatorIndividualMetrics(
        generator, model)  # Evaluator(generator, model)

    scores = evaluator.evaluate(test_dataset, args)

    model.save_scores(scores)
