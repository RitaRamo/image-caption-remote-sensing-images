import logging
from args_parser import get_args
import pandas as pd
from dataset import get_dataset, get_vocab_info
from evaluator.evaluate import Evaluator
from generators.abstract_generator import PATH
from generators.features_extracted.simple_generator import \
    FeaturesExtractedSimpleGenerator
from generators.fine_tuned.simple_generator import FineTunedSimpleGenerator
from models.simple_model import SimpleModel
from models.simple_model_finetuning import SimpleFineTunedModel

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    args = get_args()
    logging.info(args.__dict__)

    raw_dataset = pd.read_json(PATH + "dataset_rsicd.json")

    vocab_info = get_vocab_info("src/datasets/RSICD/dataset/")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    test_dataset = get_dataset(
        "src/datasets/RSICD/dataset/test.json")

    if args.fine_tuning:
        logging.info("fine tuning")
        generator = FineTunedSimpleGenerator(raw_dataset)

    else:
        logging.info("feature extraction")

        generator = FeaturesExtractedSimpleGenerator(raw_dataset)

    model_class = globals()[args.model_class_str]

    model = model_class(
        str(args.__dict__),
        vocab_size,
        max_len,
        generator.get_shape_of_input_image()
    )

    model.load()

    evaluator = Evaluator(generator, model, token_to_id, id_to_token)

    evaluator.evaluate(test_dataset)


# por o score; ver se funca
# por o score até dar no final
# continuar com o resto do train
