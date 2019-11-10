import argparse


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        '--file_name', help='name of file that was used to fill the all other arguments', default=None)

    parser.add_argument(
        '--model_class_str', help='class name of the model to train', default="SimpleModel")

    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--fine_tuning', action='store_true', default=False,
                        help='Set a switch to true')

    args = parser.parse_args()

    return args
