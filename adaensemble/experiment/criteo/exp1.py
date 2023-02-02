import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from src.io import (
    build_csv_vocabulary, build_csv_dataset,
    count_lines
)
from src.transformation import (
    gbdt_feature_engineering_fn, log_square_binning_fn,
    tf_signedlog
)
from src.moe import (
    GFTRL,
    NoisyTopKGating,
    NoisyAnnealingTopKGating,
    GumbelTopKGating,
    GumbelThresholdingGating,
    GatedDense,
    Cross,
    PolynomialInteraction,
    InteractionMachine,
    GeneralizedInteraction,
    MultiHeadSelfAttention,
    CIN,
    CNN1D,
    Sequential,
    Expert,
    SparseMoE,
    SequentialDynamicMoE,
    RankingModel
)


# OpenMP Threads
os.environ['OMP_NUM_THREADS'] = '{:d}'.format(1)


# Parameters Definition
label_name = ['Label']
integer_feature_name = ['I'+str(i) for i in range(1, 13)]
categorical_feature_name = ['C'+str(i) for i in range(1, 27)]
feature_name = integer_feature_name + categorical_feature_name

train_csv_filename = './data/criteo/csv/train.csv'
test_csv_filename = './data/criteo/csv/test.csv'
vocabulary_folder = './file/criteo/vocabulary/'
vocabulary_threshold = 20
vocabulary_running_flag = False

schema_dict = dict(
    [
        (each, [int(0)])
        for each in integer_feature_name
    ] +
    [
        (each, [''])
        for each in categorical_feature_name
    ] +
    [
        (each, [int(0)])
        for each in label_name
    ]
)


# Vocabulary
if vocabulary_running_flag:
    build_csv_vocabulary(
        filenames=[train_csv_filename],
        columns=categorical_feature_name,
        vocabulary_folder=vocabulary_folder,
        threshold=vocabulary_threshold
    )


# Input Function
def train_input_fn(batch_size, num_epochs):
    dataset = build_csv_dataset(
        filenames=[train_csv_filename],
        feature_name=feature_name,
        label_name=label_name,
        schema_dict=schema_dict,
        compression_type=None,
        buffer_size=128 * 1024 * 1024,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        shuffle=False,
        shuffle_buffer_size=1024 * 8,
        num_epochs=num_epochs,
        batch_size=batch_size,
        prefetch_buffer_size=8,
        num_parallel_calls=4,
        feature_engineering_fn=log_square_binning_fn,
        binning_cols=integer_feature_name
    )
    return dataset


def test_input_fn(batch_size):
    dataset = build_csv_dataset(
        filenames=[test_csv_filename],
        feature_name=feature_name,
        label_name=label_name,
        schema_dict=schema_dict,
        compression_type=None,
        buffer_size=128 * 1024 * 1024,
        field_delim=',',
        use_quote_delim=True,
        na_value='',
        shuffle=False,
        shuffle_buffer_size=1024 * 8,
        num_epochs=1,
        batch_size=batch_size,
        prefetch_buffer_size=8,
        num_parallel_calls=4,
        feature_engineering_fn=log_square_binning_fn,
        binning_cols=integer_feature_name
    )
    return dataset


# Feature Column
integer_feature_categorical_columns = [
    tf.feature_column.categorical_column_with_identity(
        key=col + '_' + 'bin',
        num_buckets=64
    )
    for col in integer_feature_name
]

categorical_feature_categorical_columns = [
    tf.feature_column.categorical_column_with_vocabulary_file(
        key=col,
        vocabulary_file=vocabulary_folder + str(col) + '.txt',
        num_oov_buckets=0,
        default_value=None,
        dtype=tf.string
    )
    for col in categorical_feature_name
]

integer_feature_embedding_columns = [
    tf.feature_column.embedding_column(
        categorical_column=each,
        dimension=8,
        combiner='mean',
        initializer=None,
        max_norm=None,
        trainable=True
    )
    for each in integer_feature_categorical_columns
]

categorical_feature_embedding_columns = [
    tf.feature_column.embedding_column(
        categorical_column=each,
        dimension=np.ceil(np.log2(
            count_lines(vocabulary_folder+each.key+'.txt')
        )) * 1,
        combiner='mean',
        initializer=None,
        max_norm=None,
        trainable=True
    )
    for each in categorical_feature_categorical_columns
]

embedding_columns = [
    tf.feature_column.embedding_column(each, dimension=16, combiner='sqrtn')
    for each in (
        integer_feature_categorical_columns +
        categorical_feature_categorical_columns
    )
]


# Dataset
train_dataset = train_input_fn(batch_size=1024 * 4, num_epochs=1)
test_dataset = test_input_fn(batch_size=1024 * 4)


# Model
NUM_FEATURE = len(embedding_columns)
EMBEDDING_SIZE = 16


def expert_gating_layer(num_experts, prior=None, name=None):
    gating_layer = NoisyAnnealingTopKGating(
        num_experts,
        boundary_k_pairs=[(1000, 3), (2000, 3)],
        noise_scheduler=lambda x: 0.5,
        reduction_ratio=16,
        projection_size=16,
        activation=tf.keras.activations.relu,
        alpha=1.0, prior=prior, zeta=0.01,
        name=name
    )
    return gating_layer

def exit_gating_layer(num_experts, prior=None, name=None):
    gating_layer = NoisyAnnealingTopKGating(
        num_experts,
        boundary_k_pairs=[(1000, 1), (2000, 1)],
        noise_scheduler=lambda x: 0.5,
        reduction_ratio=16,
        projection_size=16,
        activation=tf.keras.activations.relu,
        alpha=1.0, prior=prior, zeta=0.01,
        name=name
    )
    return gating_layer


def gated_dense_expert(**kwargs):
    expert = Expert(
        GatedDense(**kwargs),
        require_input_feature=True
    )
    return expert


def cross_expert(**kwargs):
    expert = Expert(
        Cross(**kwargs),
        require_input_feature=True
    )
    return expert


def polynomial_interaction_expert(**kwargs):
    expert = Expert(
        PolynomialInteraction(**kwargs),
        require_input_feature=True
    )
    return expert


def interaction_machine_expert(**kwargs):
    expert = Expert(
        Sequential([
            tf.keras.layers.Lambda(lambda x: x[1]),
            tf.keras.layers.Reshape((NUM_FEATURE, EMBEDDING_SIZE)),
            InteractionMachine(**kwargs),
            tf.keras.layers.Dense(
                units=NUM_FEATURE * EMBEDDING_SIZE,
                activation=tf.keras.activations.linear
            )
        ]),
        require_input_feature=True
    )
    return expert


def cnn1d_expert(**kwargs):
    expert = Expert(
        Sequential([
            CNN1D(**kwargs),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=NUM_FEATURE * EMBEDDING_SIZE,
                activation=tf.keras.activations.linear
            ),
            tf.keras.layers.Reshape((NUM_FEATURE * EMBEDDING_SIZE, 1))
        ]),
        require_input_feature=False
    )
    return expert


def cin_expert(**kwargs):
    expert = Expert(
        Sequential([
            tf.keras.layers.Reshape((NUM_FEATURE, EMBEDDING_SIZE)),
            CIN(**kwargs),
            tf.keras.layers.Dense(
                units=NUM_FEATURE * EMBEDDING_SIZE,
                activation=tf.keras.activations.linear
            )
        ]),
        require_input_feature=False
    )
    return expert


def multi_head_attention_expert(**kwargs):
    expert = Expert(
        Sequential([
            tf.keras.layers.Reshape((NUM_FEATURE, EMBEDDING_SIZE)),
            MultiHeadSelfAttention(**kwargs),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=NUM_FEATURE * EMBEDDING_SIZE,
                activation=tf.keras.activations.linear
            ),
            tf.keras.layers.Reshape((NUM_FEATURE * EMBEDDING_SIZE, 1))
        ]),
        require_input_feature=False
    )
    return expert


def generalized_interaction_expert(**kwargs):
    expert = Expert(
        Sequential([
            tf.keras.layers.Lambda(
                lambda x: [
                    tf.reshape(each, (-1, NUM_FEATURE, EMBEDDING_SIZE))
                    for each in x
                ]
            ),
            GeneralizedInteraction(**kwargs),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=NUM_FEATURE * EMBEDDING_SIZE,
                activation=tf.keras.activations.linear
            )
        ]),
        require_input_feature=True
    )
    return expert


def estimator(name):
    estimator = Sequential(
        layers=[
            tf.keras.layers.Lambda(function=lambda x: tf.reduce_sum(x, axis=-1)),
            tf.keras.layers.Dense(
                units=1,
                activation=tf.keras.activations.linear
            )
        ],
        name=name
    )
    return estimator


tf.keras.utils.set_random_seed(0)
input_layer = Sequential(
    layers=[
        tf.keras.layers.DenseFeatures(embedding_columns),
        tf.keras.layers.Reshape((NUM_FEATURE * EMBEDDING_SIZE, 1))
    ],
    name='embedding_layer'
)

moe_layer_0 = SparseMoE(
    gating_layer=expert_gating_layer(
        num_experts=5,
        prior=[0.2, 0.2, 0.4, 0.1, 0.1],
        name='expert_gating_layer_0'
    ),
    expert_layers=[
        gated_dense_expert(units=NUM_FEATURE * EMBEDDING_SIZE, reduction_ratio=16, activation=tf.keras.activations.relu),
        cross_expert(skip_connection=False),
        polynomial_interaction_expert(skip_connection=False),
        cnn1d_expert(units=NUM_FEATURE // 2, filters=16, kernel_size=4, pool_size=2),
        multi_head_attention_expert(hidden_size=8, activation=tf.keras.activations.gelu, num_heads=4)
    ],
    name='moe_layer_0'
)

moe_layer_1 = SparseMoE(
    gating_layer=expert_gating_layer(
        num_experts=5,
        prior=[0.2, 0.2, 0.4, 0.1, 0.1],
        name='expert_gating_layer_1'
    ),
    expert_layers=[
        gated_dense_expert(units=NUM_FEATURE * EMBEDDING_SIZE, reduction_ratio=16, activation=tf.keras.activations.relu),
        cross_expert(skip_connection=False),
        polynomial_interaction_expert(skip_connection=False),
        cnn1d_expert(units=NUM_FEATURE // 2, filters=16, kernel_size=4, pool_size=2),
        multi_head_attention_expert(hidden_size=8, activation=tf.keras.activations.gelu, num_heads=4)
    ],
    name='moe_layer_1'
)

moe_layer_2 = SparseMoE(
    gating_layer=expert_gating_layer(
        num_experts=5,
        prior=[0.2, 0.2, 0.4, 0.1, 0.1],
        name='expert_gating_layer_2'
    ),
    expert_layers=[
        gated_dense_expert(units=NUM_FEATURE * EMBEDDING_SIZE, reduction_ratio=16, activation=tf.keras.activations.relu),
        cross_expert(skip_connection=False),
        polynomial_interaction_expert(skip_connection=False),
        cnn1d_expert(units=NUM_FEATURE // 2, filters=16, kernel_size=4, pool_size=2),
        multi_head_attention_expert(hidden_size=8, activation=tf.keras.activations.gelu, num_heads=4)
    ],
    name='moe_layer_2'
)

moe_layer_3 = SparseMoE(
    gating_layer=expert_gating_layer(
        num_experts=5,
        prior=[0.2, 0.2, 0.4, 0.1, 0.1],
        name='expert_gating_layer_3'
    ),
    expert_layers=[
        gated_dense_expert(units=NUM_FEATURE * EMBEDDING_SIZE, reduction_ratio=16, activation=tf.keras.activations.relu),
        cross_expert(skip_connection=False),
        polynomial_interaction_expert(skip_connection=False),
        cnn1d_expert(units=NUM_FEATURE // 2, filters=16, kernel_size=4, pool_size=2),
        multi_head_attention_expert(hidden_size=8, activation=tf.keras.activations.gelu, num_heads=4)
    ],
    name='moe_layer_3'
)


interaction_layer = SequentialDynamicMoE(
    layers=[moe_layer_0, moe_layer_1, moe_layer_2, moe_layer_3],
    estimators=[estimator(name='estimator_' + str(i)) for i in range(4)],
    router=exit_gating_layer(
        num_experts=4,
        prior=[0.1, 0.2, 0.6, 0.1],
        name='exit_gating_layer'
    ),
    skip_connection=True,
    layer_norm=False
)

output_layer = tf.keras.layers.Lambda(
    function=lambda x: tf.math.sigmoid(x),
    name='output_layer'
)

model = RankingModel(
    input_layer,
    interaction_layer,
    output_layer
)

model.compile(
    embedding_optimizer=GFTRL(
        alpha=0.02, beta=1e-4, lambda1=0.001, lambda2=0.001
    ),
    model_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss_fn=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.AUC(name='AUC', num_thresholds=10000),
        tf.keras.metrics.BinaryCrossentropy(name='LogLoss')
    ],
    jit_compile=False
)


for _ in range(3):
    model.train(train_dataset, metric_freq=100)
    model.test(test_dataset)
