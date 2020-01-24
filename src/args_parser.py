import argparse
from preprocess_data.images import ImageNetModelsPretrained
from models.embeddings import EmbeddingsType


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

    # parser.add_argument(
    #     '--embedding_type', help='embedding type (glove,spacy or None)', default=None)

    parser.add_argument('--embedding_type', type=str, default=None,
                        choices=[model.value for model in EmbeddingsType])

    # parser.add_argument('--glove_emb_dim',
    #                     choices=(50, 100, 200, 300), default=50, type=int)

    #print("this is parser so far", parser.parse_known_args())
    opts, _ = parser.parse_known_args()
    if opts.embedding_type == EmbeddingsType.GLOVE.value:
        parser.add_argument('--embedding_size',
                            choices=(50, 100, 200, 300), default=50, type=int)
    elif opts.embedding_type == EmbeddingsType.SPACY.value:
        parser.add_argument('--embedding_size', type=int, default=None)
    else:
        parser.add_argument('--embedding_size', type=int, default=300)
    # if opts.embedding_type == EmbeddingsType.GLOVE.value:
    #     parser.add_argument('--glove_emb_dim',
    #                         choices=(50, 100, 200, 300), default=50, type=int)
    # else:
    #     parser.add_argument('--embedding_size', type=int, default=300)

    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--fine_tuning', action='store_true', default=False,
                        help='Set a switch to true')

    args = parser.parse_args()

    return args
