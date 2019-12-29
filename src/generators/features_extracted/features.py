
import tensorflow as tf


def extract_features(extractor_features_model, img):
    features = extractor_features_model(img)

    # (1, 64, 2048) -> -1 is to flatten
    features = tf.reshape(
        features,
        (
            features.shape[0],
            -1,
            features.shape[3]
        )
    )

    # [0] to remove first dim [1, 64, 2048] -> [64,2048]
    return features  # [0]
