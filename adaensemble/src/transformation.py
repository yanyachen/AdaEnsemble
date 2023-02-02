import numpy as np
import collections
import tensorflow as tf
import xgboost as xgb


def gbdt_feature_engineering_fn(features, gbdt):
    gbdt_input = tf.stack([
        features[col]
        for col in gbdt.get_info('feature')
    ], axis=1)

    def gbdt_contribs(gbdt_input):
        pred_contribs = gbdt.predict(
            data=gbdt_input,
            method='contribs'
        )
        return pred_contribs

    def gbdt_leaf(gbdt_input):
        pred_leaf = gbdt.predict(
            data=gbdt_input,
            method='leaf'
        )
        return pred_leaf

    gbdt_output_contribs = tf.py_func(
        gbdt_contribs, [gbdt_input], tf.float32
    )
    gbdt_output_leaf = tf.py_func(
        gbdt_leaf, [gbdt_input], tf.int32
    )

    gbdt_output_dict = collections.OrderedDict()
    for index, name in enumerate(gbdt.get_info('contribs')):
        gbdt_output_dict[name] = gbdt_output_contribs[:, index]

    for index, name in enumerate(gbdt.get_info('leaf').keys()):
        gbdt_output_dict[name] = gbdt_output_leaf[:, index]

    return gbdt_output_dict


def log_square_binning_fn(features, binning_cols):
    binning_output_dict = collections.OrderedDict()
    for col in binning_cols:
        new_col = col + '_' + 'bin'
        feature = tf.cast(features[col], dtype=tf.float32)
        new_feature = tf.where(
            tf.compat.v1.debugging.is_nan(feature),
            -1.0 * tf.ones_like(feature),
            tf.math.floor(tf.math.log(feature ** 2 + 1))
        )
        new_feature = tf.cast(new_feature, dtype=tf.int32)
        binning_output_dict[new_col] = new_feature
    return binning_output_dict


def pred_regularizing(y_pred, y_true_mean, eps):
    pred_reg = (1.0 - eps) * y_pred + eps * y_true_mean
    return pred_reg


def signedlog(x):
    return (
        np.sign(x) * np.log(1.0 + np.sign(x) * x)
    )


def tf_signedlog(x):
    result = tf.where(
            tf.math.greater_equal(x, 0),
            +1.0 * tf.math.log(1.0 + x),
            -1.0 * tf.math.log(1.0 - x)
        )
    return result
