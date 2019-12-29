import argparse
from preprocess_data.images import ImageNetModelsPretrained


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        '--file_name', help='name of file that was used to fill the all other arguments', default=None)

    parser.add_argument(
        '--model_class_str', help='class name of the model to train', default="SimpleModel")

    parser.add_argument('--image_model_type', type=str, default=ImageNetModelsPretrained.INCEPTION_V3.value,
                        choices=[model.value for model in ImageNetModelsPretrained])

    parser.add_argument('--epochs', type=int, default=33,
                        help='define epochs to train the model')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='define batch size to train the model')

    parser.add_argument('--disable_steps', action='store_true', default=False,
                        help='Conf just for testing: make the model run only 1 steps instead of the steps that was supposed')

    parser.add_argument('--disable_metrics', action='store_true', default=False,
                        help='Conf just for testing: make the model does not run the metrics')

    parser.add_argument(
        '--embedding_type', help='embedding type (glove,spacy or None)', default=None)

    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--fine_tuning', action='store_true', default=False,
                        help='Set a switch to true')

    args = parser.parse_args()

    return args
