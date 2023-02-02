import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa


class DenseDispatcher2D(object):

    def __init__(self, expert_gates, weighting_step):
        assert weighting_step in ('pre', 'post')
        self.expert_gates = expert_gates
        self.weighting_step = weighting_step

        self.batch_size = tf.shape(expert_gates)[0]
        self.num_experts = expert_gates.shape[1]

    def dispatch(self, inputs):
        if self.weighting_step == 'pre':
            expert_gates = self.expert_gates
        elif self.weighting_step == 'post':
            expert_gates = tf.cast(self.expert_gates > 0.0, tf.float32)

        inputs_all_expert = tf.einsum(
            'np..., nk -> nkp...',
            inputs,
            expert_gates
        )
        inputs_all_expert_splitted = tf.split(
            value=inputs_all_expert,
            num_or_size_splits=self.num_experts,
            axis=1
        )
        inputs_per_expert = [
            tf.squeeze(x, axis=1)
            for x in inputs_all_expert_splitted
        ]
        return inputs_per_expert

    def combine(self, expert_outputs):
        if self.weighting_step == 'pre':
            expert_gates = tf.cast(self.expert_gates > 0.0, tf.float32)
        elif self.weighting_step == 'post':
            expert_gates = self.expert_gates

        expert_outputs_stack = tf.stack(expert_outputs, axis=1)
        expert_outputs_combined = tf.einsum(
            'nkp..., nk -> np...',
            expert_outputs_stack,
            expert_gates
        )
        return expert_outputs_combined


class SparseDispatcher2D(object):

    def __init__(self, expert_gates):
        self.expert_gates = expert_gates
        self.batch_size = tf.shape(expert_gates)[0]
        self.num_experts = expert_gates.shape[1]
        self.size_per_expert = tf.math.reduce_sum(
            tf.cast(expert_gates > 0.0, tf.int64),
            axis=0
        )

        nonzero_indices = tf.cast(
            tf.where(
                tf.transpose(expert_gates, perm=[1, 0]) > 0.0
            ),
            tf.int64
        )
        self.expert_index, self.batch_index = tf.unstack(
            nonzero_indices, num=2, axis=1
        )

        self.nonzero_gates = tf.gather(
            tf.reshape(expert_gates, shape=(-1, 1)),
            self.batch_index * self.num_experts +
            self.expert_index
        )

    def dispatch(self, inputs):
        inputs_gathered = tf.gather(
            params=inputs,
            indices=self.batch_index,
            axis=0,
            batch_dims=0
        )
        inputs_per_expert = tf.split(
            value=inputs_gathered,
            num_or_size_splits=self.size_per_expert,
            axis=0,
            num=self.num_experts
        )
        return inputs_per_expert

    def combine(self, expert_outputs, weight_by_gates=False):
        expert_outputs_concat = tf.concat(expert_outputs, axis=0)
        if weight_by_gates:
            expert_outputs_concat = tf.einsum(
                'b..., b... -> b...',
                expert_outputs_concat, self.nonzero_gates
            )
        expert_outputs_combined = tf.math.unsorted_segment_sum(
            expert_outputs_concat,
            segment_ids=self.batch_index,
            num_segments=self.batch_size
        )
        return expert_outputs_combined


class ExpertRouter(tf.keras.layers.Layer):

    def __init__(
        self,
        num_experts,
        reduction_ratio,
        projection_size,
        activation,
        learnable_temperature=True,
        **kwargs
    ):
        super(ExpertRouter, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.reduction_ratio = reduction_ratio
        self.projection_size = projection_size
        self.activation = activation
        self.learnable_temperature = learnable_temperature

    def build(self, input_shape):
        num_logits = np.prod(input_shape[1:])
        self.projection_dnn = Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    units=int(num_logits // self.reduction_ratio),
                    activation=self.activation,
                    use_bias=True
                ),
                tf.keras.layers.Dense(
                    units=self.projection_size,
                    activation=tf.keras.activations.linear,
                    use_bias=True
                )
            ],
            name='projection_network'
        )

        initializer = tf.keras.initializers.GlorotUniform()
        self.expert_embedding = tf.Variable(
            initial_value=initializer(
                shape=(self.num_experts, self.projection_size),
                dtype=tf.float32
            ),
            trainable=True,
            name='expert_embedding',
            dtype=tf.float32
        )

        self.temperature = tf.Variable(
            initial_value=1.0,
            trainable=self.learnable_temperature,
            dtype=tf.float32,
            name='temperature'
        )

    def call(self, inputs, training=False):
        sample_embedding = self.projection_dnn(inputs, training)
        routing_logits = tf.einsum(
            'bk, ek -> be',
            tf.math.l2_normalize(sample_embedding, axis=1),
            tf.math.l2_normalize(self.expert_embedding, axis=1)
        )
        routing_logits /= self.temperature
        return routing_logits


class TopKSoftMax(tf.keras.layers.Layer):

    def __init__(
        self,
        k=1,
        **kwargs
    ):
        super(TopKSoftMax, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.depth = input_shape[1]

    def call(self, inputs):
        top_k_values, top_k_indices = tf.math.top_k(
            inputs,
            k=self.k,
            sorted=False
        )
        top_k_one_hot = tf.math.reduce_sum(
            tf.one_hot(
                indices=top_k_indices,
                depth=self.depth,
                axis=-1
            ),
            axis=1
        )

        sparse_inputs = tf.where(
            top_k_one_hot > 0.0,
            inputs,
            tf.ones_like(inputs) * -1e+16
        )
        sparse_gates = tf.math.softmax(
            sparse_inputs,
            axis=1
        )

        return sparse_gates


class AnnealingTopKSoftMax(tf.keras.layers.Layer):

    def __init__(
        self,
        boundary_k_pairs,
        **kwargs
    ):
        super(AnnealingTopKSoftMax, self).__init__(**kwargs)
        self.boundary_k_pairs = boundary_k_pairs
        self.num_calls = tf.Variable(
            initial_value=0.0, trainable=False, name='num_calls'
        )
        self.default_k = boundary_k_pairs[-1][1]

    def generate_pred_fn_pairs(self):
        pred_k_pairs = []
        for i in range(len(self.boundary_k_pairs)):
            if i == 0:
                left_boundary = 0
            else:
                left_boundary = self.boundary_k_pairs[i - 1][0]
            right_boundary = self.boundary_k_pairs[i][0]
            k = self.boundary_k_pairs[i][1]

            pred_k_pairs.append(
                (
                    tf.math.logical_and(
                        self.num_calls >= left_boundary,
                        self.num_calls < right_boundary,
                    ),
                    lambda k=k: k
                )
            )
        return pred_k_pairs

    def build(self, input_shape):
        self.depth = input_shape[1]

    def call(self, inputs, training=False):
        if training:
            assign_op = self.num_calls.assign_add(1.0, read_value=False)
            preconditions = [] if assign_op is None else [assign_op]
            with tf.control_dependencies(preconditions):
                k = tf.case(
                    pred_fn_pairs=self.generate_pred_fn_pairs(),
                    default=lambda: self.default_k,
                    exclusive=True,
                    strict=False
                )
        else:
            k = tf.case(
                pred_fn_pairs=self.generate_pred_fn_pairs(),
                default=lambda: self.default_k,
                exclusive=False,
                strict=False
            )

        top_k_values, top_k_indices = tf.math.top_k(
            inputs,
            k=k,
            sorted=False
        )
        top_k_one_hot = tf.math.reduce_sum(
            tf.one_hot(
                indices=top_k_indices,
                depth=self.depth,
                axis=-1
            ),
            axis=1
        )

        sparse_inputs = tf.where(
            top_k_one_hot > 0.0,
            inputs,
            tf.ones_like(inputs) * -1e+16
        )
        sparse_gates = tf.math.softmax(
            sparse_inputs,
            axis=1
        )

        return sparse_gates


class GumbelTopKSoftMax(tf.keras.layers.Layer):

    def __init__(
        self,
        temperature_scheduler=lambda x: 1.0,
        k=1,
        **kwargs
    ):
        super(GumbelTopKSoftMax, self).__init__(**kwargs)
        self.temperature_scheduler = temperature_scheduler
        self.k = k
        self.num_calls = tf.Variable(
            initial_value=0.0, trainable=False, name='num_calls'
        )

    def build(self, input_shape):
        self.depth = input_shape[1]

    def sample_gumbel(self, shape):
        uniform_samples = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        outputs = -tf.math.log(-tf.math.log(uniform_samples + 1e-8) + 1e-8)
        return outputs

    def call(self, inputs, training=False):
        if training:
            assign_op = self.num_calls.assign_add(1.0, read_value=False)
            preconditions = [] if assign_op is None else [assign_op]
            with tf.control_dependencies(preconditions):
                temperature = self.temperature_scheduler(self.num_calls)
                noise = self.sample_gumbel(inputs.shape)
        else:
            temperature = self.temperature_scheduler(self.num_calls)
            noise = 0.0

        probs = tf.math.softmax(
            tf.math.divide(
                inputs + noise,
                temperature
            ),
            axis=1
        )

        top_k_values, top_k_indices = tf.math.top_k(
            probs,
            k=self.k,
            sorted=False
        )
        top_k_one_hot = tf.math.reduce_sum(
            tf.one_hot(
                indices=top_k_indices,
                depth=self.depth,
                axis=-1
            ),
            axis=1
        )

        sparse_inputs = tf.where(
            top_k_one_hot > 0.0,
            inputs,
            tf.ones_like(inputs) * -1e+16
        )
        sparse_gates = tf.math.softmax(
            sparse_inputs,
            axis=1
        )

        return sparse_gates


class GumbelSoftMaxThresholding(tf.keras.layers.Layer):

    def __init__(
        self,
        temperature_scheduler=lambda x: 1.0,
        threshold_scheduler=lambda x: 0.05,
        **kwargs
    ):
        super(GumbelSoftMaxThresholding, self).__init__(**kwargs)
        self.temperature_scheduler = temperature_scheduler
        self.threshold_scheduler = threshold_scheduler
        self.num_calls = tf.Variable(
            initial_value=0.0, trainable=False, name='num_calls'
        )

    def sample_gumbel(self, shape):
        uniform_samples = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        outputs = -tf.math.log(-tf.math.log(uniform_samples + 1e-8) + 1e-8)
        return outputs

    def call(self, inputs, training=False):
        if training:
            assign_op = self.num_calls.assign_add(1.0, read_value=False)
            preconditions = [] if assign_op is None else [assign_op]
            with tf.control_dependencies(preconditions):
                temperature = self.temperature_scheduler(self.num_calls)
                threshold = self.threshold_scheduler(self.num_calls)
                noise = self.sample_gumbel(inputs.shape)
        else:
            temperature = self.temperature_scheduler(self.num_calls)
            threshold = self.threshold_scheduler(self.num_calls)
            noise = 0.0

        probs = tf.math.softmax(
            tf.math.divide(
                inputs + noise,
                temperature
            ),
            axis=1
        )

        sparse_inputs = tf.where(
            probs > threshold,
            inputs,
            tf.ones_like(inputs) * -1e+16
        )
        sparse_gates = tf.math.softmax(
            sparse_inputs,
            axis=1
        )

        return sparse_gates


class LoadBalanceLoss(tf.keras.layers.Layer):

    def __init__(
        self,
        alpha=0.01,
        prior=None,
        **kwargs
    ):
        super(LoadBalanceLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.prior = prior

    def build(self, input_shape):
        self.num_experts = input_shape[0][1]
        if self.prior:
            assert self.num_experts == len(self.prior)
            self.prior_weight = tf.math.divide(
                1.0,
                (self.num_experts * tf.constant(self.prior, dtype=tf.float32))
            )
        else:
            self.prior_weight = tf.repeat(1.0, self.num_experts)

    def call(self, inputs):
        raw_logits, expert_probs = inputs
        raw_probs = tf.math.softmax(raw_logits, axis=-1)

        dispatched_fraction = tf.reduce_mean(
            tf.cast(expert_probs > 0.0, tf.float32),
            axis=0
        )

        router_probability = tf.reduce_mean(
            raw_probs,
            axis=0
        )

        load_balance_loss = self.alpha * (self.num_experts ** 2) *\
            tf.reduce_mean(
                tf.math.multiply(
                    dispatched_fraction * router_probability,
                    self.prior_weight
                )
            )
        # load_balance_loss = tf.reduce_sum(dispatched_fraction * router_probability / self.prior)

        return load_balance_loss


class RouterZLoss(tf.keras.layers.Layer):

    def __init__(
        self,
        zeta=0.001,
        **kwargs
    ):
        super(RouterZLoss, self).__init__(**kwargs)
        self.zeta = zeta

    def call(self, inputs):
        log_z = tf.math.reduce_logsumexp(inputs, axis=1)
        router_z_loss_vec = tf.square(log_z)
        router_z_loss = tf.reduce_mean(router_z_loss_vec)

        return router_z_loss


class RandomNoise(tf.keras.layers.Layer):

    def __init__(
        self,
        noise_scheduler=lambda x: 0.01,
        **kwargs
    ):
        super(RandomNoise, self).__init__(**kwargs)
        self.noise_scheduler = noise_scheduler
        self.num_calls = tf.Variable(
            initial_value=0.0, trainable=False, name='num_calls'
        )

    def call(self, inputs, training=False):
        if not training:
            return inputs

        assign_op = self.num_calls.assign_add(1.0, read_value=False)
        preconditions = [] if assign_op is None else [assign_op]
        with tf.control_dependencies(preconditions):
            epsilon = self.noise_scheduler(self.num_calls)
            noisy_inputs = inputs * tf.random.uniform(
                shape=tf.shape(inputs),
                minval=1.0 - epsilon,
                maxval=1.0 + epsilon,
                dtype=inputs.dtype
            )

        return noisy_inputs


class NoisyTopKGating(tf.keras.layers.Layer):

    def __init__(
        self,
        num_experts,
        k=1,
        noise_scheduler=lambda x: 0.01,
        reduction_ratio=16,
        projection_size=16,
        activation=tf.keras.activations.relu,
        alpha=0.01, prior=None, zeta=0.001,
        **kwargs
    ):
        super(NoisyTopKGating, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.k = k
        self.noise_scheduler = noise_scheduler
        self.reduction_ratio = reduction_ratio
        self.projection_size = projection_size
        self.activation = activation
        self.alpha = alpha
        self.prior = prior
        self.zeta = zeta

        self.gating_dnn = tf.keras.Sequential(
            [
                RandomNoise(noise_scheduler=self.noise_scheduler),
                ExpertRouter(
                    num_experts=self.num_experts,
                    reduction_ratio=self.reduction_ratio,
                    projection_size=self.projection_size,
                    activation=self.activation
                )
            ]
        )
        self.top_k_softmax = TopKSoftMax(k=self.k)
        self.load_balance_loss = LoadBalanceLoss(
            alpha=self.alpha, prior=self.prior
        )
        self.router_z_loss = RouterZLoss(zeta=self.zeta)

    def call(self, inputs, training=False):
        gating_logits = self.gating_dnn(inputs, training)
        gating_probs = self.top_k_softmax(gating_logits)
        if training:
            self.add_loss(self.load_balance_loss([gating_logits, gating_probs]))
            self.add_loss(self.router_z_loss(gating_logits))
        return gating_probs


class NoisyAnnealingTopKGating(tf.keras.layers.Layer):

    def __init__(
        self,
        num_experts,
        boundary_k_pairs,
        noise_scheduler=lambda x: 0.01,
        reduction_ratio=16,
        projection_size=16,
        activation=tf.keras.activations.relu,
        alpha=0.01, prior=None, zeta=0.001,
        **kwargs
    ):
        super(NoisyAnnealingTopKGating, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.boundary_k_pairs = boundary_k_pairs
        self.noise_scheduler = noise_scheduler
        self.reduction_ratio = reduction_ratio
        self.projection_size = projection_size
        self.activation = activation
        self.alpha = alpha
        self.prior = prior
        self.zeta = zeta

        self.gating_dnn = tf.keras.Sequential(
            [
                RandomNoise(noise_scheduler=self.noise_scheduler),
                ExpertRouter(
                    num_experts=self.num_experts,
                    reduction_ratio=self.reduction_ratio,
                    projection_size=self.projection_size,
                    activation=self.activation
                )
            ]
        )
        self.top_k_softmax = AnnealingTopKSoftMax(
            boundary_k_pairs=self.boundary_k_pairs
        )
        self.load_balance_loss = LoadBalanceLoss(
            alpha=self.alpha, prior=self.prior
        )
        self.router_z_loss = RouterZLoss(zeta=self.zeta)

    def call(self, inputs, training=False):
        gating_logits = self.gating_dnn(inputs, training)
        gating_probs = self.top_k_softmax(gating_logits)
        if training:
            self.add_loss(self.load_balance_loss([gating_logits, gating_probs]))
            self.add_loss(self.router_z_loss(gating_logits))
        return gating_probs


class GumbelTopKGating(tf.keras.layers.Layer):

    def __init__(
        self,
        num_experts,
        temperature_scheduler=lambda x: 1.0,
        k=1,
        reduction_ratio=16,
        projection_size=16,
        activation=tf.keras.activations.relu,
        alpha=0.01, prior=None, zeta=0.001,
        **kwargs
    ):
        super(GumbelTopKGating, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.temperature_scheduler = temperature_scheduler
        self.k = k
        self.reduction_ratio = reduction_ratio
        self.projection_size = projection_size
        self.activation = activation
        self.alpha = alpha
        self.prior = prior
        self.zeta = zeta

        self.gating_dnn = ExpertRouter(
            num_experts=self.num_experts,
            reduction_ratio=self.reduction_ratio,
            projection_size=self.projection_size,
            activation=self.activation
        )
        self.gumbel_softmax = GumbelTopKSoftMax(
            temperature_scheduler=self.temperature_scheduler,
            k=self.k
        )
        self.load_balance_loss = LoadBalanceLoss(
            alpha=self.alpha, prior=self.prior
        )
        self.router_z_loss = RouterZLoss(zeta=self.zeta)

    def call(self, inputs, training=False):
        gating_logits = self.gating_dnn(inputs, training)
        gating_probs = self.gumbel_softmax(gating_logits, training)
        if training:
            self.add_loss(self.load_balance_loss([gating_logits, gating_probs]))
            self.add_loss(self.router_z_loss(gating_logits))
        return gating_probs


class GumbelThresholdingGating(tf.keras.layers.Layer):

    def __init__(
        self,
        num_experts,
        temperature_scheduler=lambda x: 1.0,
        threshold_scheduler=lambda x: 0.05,
        reduction_ratio=16,
        projection_size=16,
        activation=tf.keras.activations.relu,
        alpha=0.01, prior=None, zeta=0.001,
        **kwargs
    ):
        super(GumbelThresholdingGating, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.temperature_scheduler = temperature_scheduler
        self.threshold_scheduler = threshold_scheduler
        self.reduction_ratio = reduction_ratio
        self.projection_size = projection_size
        self.activation = activation
        self.alpha = alpha
        self.prior = prior
        self.zeta = zeta

        self.gating_dnn = ExpertRouter(
            num_experts=self.num_experts,
            reduction_ratio=self.reduction_ratio,
            projection_size=self.projection_size,
            activation=self.activation
        )
        self.gumbel_softmax = GumbelSoftMaxThresholding(
            temperature_scheduler=self.temperature_scheduler,
            threshold_scheduler=self.threshold_scheduler
        )
        self.load_balance_loss = LoadBalanceLoss(
            alpha=self.alpha, prior=self.prior
        )
        self.router_z_loss = RouterZLoss(zeta=self.zeta)

    def call(self, inputs, training=False):
        gating_logits = self.gating_dnn(inputs, training)
        gating_probs = self.gumbel_softmax(gating_logits, training)
        if training:
            self.add_loss(self.load_balance_loss([gating_logits, gating_probs]))
            self.add_loss(self.router_z_loss(gating_logits))
        return gating_probs


class SmoothStep(tf.keras.layers.Layer):

    def __init__(
        self,
        gamma=1.0,
        **kwargs
    ):
        super(SmoothStep, self).__init__(**kwargs)
        self._lower_bound = -gamma / 2.0
        self._upper_bound = gamma / 2.0
        self._a3 = -2 / (gamma ** 3.0)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def call(self, inputs):
        return tf.where(
            inputs <= self._lower_bound,
            tf.zeros_like(inputs),
            tf.where(
                inputs >= self._upper_bound,
                tf.ones_like(inputs),
                self._a3 * (inputs**3) + self._a1 * inputs + self._a0
            )
        )


class EntropyRegularizer(tf.keras.layers.Layer):

    def __init__(
        self,
        entropy_scheduler=lambda x: 1e-6,
        **kwargs
    ):
        super(EntropyRegularizer, self).__init__(**kwargs)
        self.num_calls = tf.Variable(
            initial_value=0.0, trainable=False, name='num_calls'
        )
        self.entropy_scheduler = entropy_scheduler

    def call(self, inputs):
        assign_op = self.num_calls.assign_add(1.0, read_value=False)
        preconditions = [] if assign_op is None else [assign_op]
        with tf.control_dependencies(preconditions):
            reg_rate = self.entropy_scheduler(self.num_calls)
            entropy = -tf.math.reduce_mean(inputs * tf.math.log(inputs + 1e-6))
            return reg_rate * entropy


class DSelectKGating(tf.keras.layers.Layer):

    def __init__(
        self,
        num_experts,
        k=1,
        reduction_ratio=16, activation=tf.keras.activations.relu,
        gamma=1.0,
        entropy_scheduler=lambda x: 1e-6,
        **kwargs
    ):
        super(DSelectKGating, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.k = k
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.gamma = gamma
        self.entropy_scheduler = entropy_scheduler

        self.smooth_step = SmoothStep(gamma=self.gamma)
        self.entropy_reg = EntropyRegularizer(
            entropy_scheduler=self.entropy_scheduler
        )

    def build(self, input_shape):
        num_logits = input_shape[1]
        self.num_binary = math.ceil(math.log2(self.num_experts))
        self.power_of_2_flag = (self.num_experts == 2 ** self.num_binary)

        self.z_dnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=int(num_logits // self.reduction_ratio),
                    activation=self.activation,
                    use_bias=True
                ),
                tf.keras.layers.Dense(
                    units=self.k * self.num_binary,
                    activation=tf.keras.activations.linear,
                    use_bias=True
                )
            ]
        )
        self.w_dnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=int(num_logits // self.reduction_ratio),
                    activation=self.activation,
                    use_bias=True
                ),
                tf.keras.layers.Dense(
                    units=self.k,
                    activation=tf.keras.activations.linear,
                    use_bias=True
                )
            ]
        )

        binary_matrix = np.array([
            list(np.binary_repr(i, width=self.num_binary))
            for i in range(self.num_experts)
        ]).astype(bool)
        self.binary_codes = tf.expand_dims(
            tf.constant(binary_matrix, dtype=bool),
            axis=0
        )

    def _add_regularization_loss(self, selector_outputs):
        self.add_loss(self.entropy_reg(selector_outputs))

        if not self.power_of_2_flag:
            self.add_loss(
                tf.math.reduce_sum(
                    1.0 / tf.math.reduce_sum(selector_outputs, axis=-1)
                )
            )

    def call(self, inputs, training=False):
        # Weighted Neural Oblivious Decision Ensembles

        # Shape = (batch_size, k, 1, num_binary)
        sample_logits = tf.reshape(
            self.z_dnn(inputs, training),
            shape=[-1, self.k, 1, self.num_binary]
        )
        sample_activations = self.smooth_step(sample_logits)

        # Shape = (batch_size, k, num_experts)
        selector_outputs = tf.math.reduce_prod(
            tf.where(
                tf.expand_dims(self.binary_codes, axis=0),
                sample_activations,
                1.0 - sample_activations
            ),
            axis=3
        )

        # Shape = (batch_size, k, 1)
        selector_weights = tf.nn.softmax(
            tf.expand_dims(
                self.w_dnn(inputs, training),
                axis=2
            ),
            axis=1
        )

        # Shape = (batch_size, num_experts)
        expert_weights = tf.math.reduce_sum(
            selector_weights * selector_outputs,
            axis=1
        )

        # Regularization
        if training:
            self._add_regularization_loss(selector_outputs)

        return expert_weights


class SubSpaceSplit(tf.keras.layers.Layer):

    def __init__(
        self,
        num_sub_spaces,
        **kwargs
    ):
        super(SubSpaceSplit, self).__init__(**kwargs)
        self.num_sub_spaces = num_sub_spaces

    def call(self, inputs, training=False):
        outputs = tf.concat(
            tf.split(inputs, self.num_sub_spaces, axis=2),
            axis=1
        )
        return outputs


class EmbedDense(tf.keras.layers.Layer):

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        **kwargs
    ):
        super(EmbedDense, self).__init__(**kwargs)
        self.dense_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.permute_layer = tf.keras.layers.Permute(
            dims=(2, 1)
        )

    def call(self, inputs, training=False):
        logits = self.permute_layer(inputs)
        logits = self.dense_layer(logits)
        outputs = self.permute_layer(logits)
        return outputs


class GatedDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        reduction_ratio,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        kernel_regularizer=None,
        **kwargs
    ):
        super(GatedDense, self).__init__(**kwargs)
        self.units = units
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        num_feature = input_shape[0][1]
        num_logits = np.prod(input_shape[0][1:])

        self.dense_layer = EmbedDense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

        self.gating_dnn = Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    units=int(num_logits // self.reduction_ratio),
                    activation=self.activation,
                    use_bias=self.use_bias
                ),
                tf.keras.layers.Dense(
                    units=num_feature,
                    activation=tf.keras.activations.sigmoid,
                    use_bias=self.use_bias
                ),
                tf.keras.layers.Lambda(
                    function=lambda x: x * 2.0
                )
            ]
        )

    def call(self, inputs, training=False):
        dense_inputs, gating_inputs = inputs
        outputs = self.dense_layer(dense_inputs)
        gates = self.gating_dnn(gating_inputs, training)
        outputs = tf.einsum(
            'bk..., bk -> bk...',
            outputs, gates
        )
        return outputs


class Cross(tf.keras.layers.Layer):

    def __init__(
        self,
        skip_connection=True,
        diag_scale=0.0,
        use_bias=True,
        kernel_initializer=None,
        kernel_regularizer=None,
        **kwargs
    ):
        super(Cross, self).__init__(**kwargs)
        self.skip_connection = skip_connection
        self.diag_scale = diag_scale
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        num_logits = input_shape[1][1]

        self.dense_layer = EmbedDense(
            units=num_logits,
            activation=tf.keras.activations.linear,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

    def call(self, inputs, training=False):
        xl, x0 = inputs

        weighted_xl = self.dense_layer(xl)

        if self.diag_scale:
            weighted_xl += self.diag_scale * xl

        xl_1 = x0 * weighted_xl

        if self.skip_connection:
            xl_1 += xl

        return xl_1


class PolynomialInteraction(tf.keras.layers.Layer):

    def __init__(
        self,
        skip_connection=True,
        diag_scale=0.0,
        use_bias=False,
        kernel_initializer=None,
        kernel_regularizer=None,
        **kwargs
    ):
        super(PolynomialInteraction, self).__init__(**kwargs)
        self.skip_connection = skip_connection
        self.diag_scale = diag_scale
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        num_logits = input_shape[0][1]

        self.dense_layer = EmbedDense(
            units=num_logits,
            activation=tf.keras.activations.linear,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

    def call(self, inputs, training=False):
        xl, x0 = inputs

        weighted_x0 = self.dense_layer(x0)

        if self.diag_scale:
            weighted_x0 += self.diag_scale * x0

        xl_1 = xl * weighted_x0

        if self.skip_connection:
            xl_1 += xl

        return xl_1


class InteractionMachine(tf.keras.layers.Layer):

    def __init__(
        self,
        order=2,
        batch_norm=False,
        **kwargs
    ):
        super(InteractionMachine, self).__init__(**kwargs)
        self.order = order
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def interact(self, inputs):
        order = len(inputs)
        if order == 1:
            p1 = inputs[0]
            outputs = p1
        elif order == 2:
            p1, p2 = inputs
            outputs = (tf.math.pow(p1, 2) - p2) / 2
        elif order == 3:
            p1, p2, p3 = inputs
            outputs = (
                tf.math.pow(p1, 3) -
                3 * p1 * p2 +
                2 * p3
            ) / 6
        elif order == 4:
            p1, p2, p3, p4 = inputs
            outputs = (
                tf.math.pow(p1, 4) -
                6 * tf.math.pow(p1, 2) * p2 +
                3 * tf.math.pow(p2, 2) +
                8 * p1 * p3 - 6 * p4
            ) / 24
        elif order == 5:
            p1, p2, p3, p4, p5 = inputs
            outputs = (
                tf.math.pow(p1, 5) -
                10 * tf.math.pow(p1, 3) * p2 +
                20 * tf.math.pow(p1, 2) * p3 -
                30 * p1 * p4 -
                20 * p2 * p3 +
                15 * p1 * tf.math.pow(p2, 2) +
                24 * p5
            ) / 120

        return outputs

    def call(self, inputs, training=False):
        inputs_power = 1.0
        inputs_power_sum_list = []
        outputs_list = []
        for _ in range(self.order):
            inputs_power *= inputs
            inputs_power_sum = tf.math.reduce_sum(inputs_power, axis=1)
            inputs_power_sum_list.append(inputs_power_sum)
            outputs_list.append(self.interact(inputs_power_sum_list))

        outputs = tf.concat(outputs_list, axis=1)
        if self.batch_norm:
            outputs = self.batch_norm_layer(outputs, training)
        return outputs       


class GeneralizedInteraction(tf.keras.layers.Layer):

    def __init__(
        self,
        num_subspaces,
        **kwargs
    ):
        super(GeneralizedInteraction, self).__init__(**kwargs)
        self.num_subspaces = num_subspaces

    def build(self, input_shape):
        self.input_subspaces = input_shape[0][1]
        self.num_fields = input_shape[1][1]
        self.embedding_size = input_shape[1][2]

        self.w = tf.Variable(
            initial_value=tf.repeat(
                tf.expand_dims(
                    tf.eye(self.embedding_size, self.embedding_size),
                    axis=0
                ),
                repeats=self.num_subspaces,
                axis=0
            ),
            trainable=True,
            dtype=tf.float32
        )
        self.alpha = tf.Variable(
            initial_value=tf.ones(
                shape=(
                    self.input_subspaces * self.num_fields,
                    self.num_subspaces
                )
            ),
            trainable=True,
            dtype=tf.float32
        )
        self.h = tf.Variable(
            initial_value=tf.ones(
                shape=(self.num_subspaces, self.embedding_size, 1)
            ),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, training=False):
        xl, x0 = inputs
        batch_size = tf.shape(xl)[0]

        outer_product = tf.einsum(
            'bnh,bnd->bnhd',
            tf.repeat(x0, repeats=self.input_subspaces, axis=1),
            tf.reshape(
                tf.repeat(xl, repeats=self.num_fields, axis=2),
                shape=(batch_size, -1, self.embedding_size)
            )
        )
        fusion = tf.linalg.matmul(
            tf.transpose(outer_product, perm=(0, 2, 3, 1)),
            self.alpha
        )
        fusion = tf.math.multiply(
            self.w,
            tf.transpose(fusion, perm=(0, 3, 1, 2))
        )
        xl_1 = tf.squeeze(
            tf.linalg.matmul(fusion, self.h),
            axis=-1
        )
        return xl_1


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        hidden_size,
        activation,
        num_heads,
        dropout,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.activation = activation
        self.num_heads = num_heads
        self.dropout = dropout

        self.query_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=self.activation
        )
        self.key_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=self.activation
        )
        self.value_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=self.activation
        )
        self.output_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=self.activation
        )
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=float(dropout))

    def call(self, inputs, training=False):
        # Input
        query, key, value = inputs
        # QKV Linear Projections
        query = self.query_projection_layer(query)
        key = self.key_projection_layer(key)
        value = self.value_projection_layer(value)

        # Multi-Head Split
        query = tf.concat(tf.split(query, self.num_heads, axis=2), axis=0)
        key = tf.concat(tf.split(key, self.num_heads, axis=2), axis=0)
        value = tf.concat(tf.split(value, self.num_heads, axis=2), axis=0)

        # Attnetion
        attention_weights = tf.einsum(
            'bpk, bqk -> bpq',
            query, key
        )

        depth = (self.hidden_size // self.num_heads)
        attention_softmax = tf.nn.softmax(
            attention_weights / tf.sqrt(float(depth)),
            axis=2
        )
        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        attention_units = tf.einsum(
            'bpq, bqk -> bpk',
            attention_softmax, value
        )

        # Multi-Head Combine
        attention_units = tf.concat(
            tf.split(attention_units, self.num_heads, axis=0), axis=2
        )

        # Output Linear Projection
        attention_units = self.output_projection_layer(attention_units)

        return attention_units


class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        hidden_size,
        activation,
        num_heads,
        dropout=0.0,
        **kwargs
    ):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.activation = activation
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention_layer = MultiHeadAttention(
            hidden_size=self.hidden_size,
            activation=self.activation,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

    def call(self, inputs, training=False):
        outputs = self.attention_layer([inputs, inputs, inputs], training)

        return outputs


class CIN(tf.keras.layers.Layer):

    def __init__(
        self,
        units,
        activation=tf.keras.activations.linear,
        skip_connection=False,
        **kwargs
    ):
        super(CIN, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.skip_connection = skip_connection

    def build(
        self,
        input_shape
    ):
        # Prep
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[2]

        self.layer_field_num = []
        self.filter = []
        self.bias = []

        # Hidden_Layer
        self.layer_field_num.append(self.field_size)
        for layer_id, num_units in enumerate(self.units):
            # Filter
            self.filter.append(
                tf.Variable(
                    tf.keras.initializers.GlorotUniform()(
                        shape=(
                            1,
                            self.layer_field_num[0] *
                            self.layer_field_num[layer_id],
                            num_units
                        )
                    ),
                    trainable=True,
                    dtype=tf.float32
                )
            )
            # Bias
            self.bias.append(
                tf.Variable(
                    tf.zeros(shape=(num_units)),
                    trainable=True,
                    dtype=tf.float32
                )
            )
            # Skip
            if not self.skip_connection:
                self.layer_field_num.append(num_units)
            else:
                self.layer_field_num.append(num_units // 2)

    def call(
        self,
        inputs,
        training
    ):
        # Prep
        cin_state_list = []
        output_list = []
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, self.field_size, self.embedding_size)
        )
        # CIN
        state_split_0 = tf.split(
            inputs, [1] * self.embedding_size, axis=2
        )
        cin_state_list.append(inputs)
        for layer_id, num_units in enumerate(self.units):
            # Z
            state_split_k = tf.split(
                cin_state_list[layer_id], [1] * self.embedding_size, axis=2
            )
            state_z_k = tf.transpose(
                a=tf.reshape(
                    tf.matmul(state_split_0, state_split_k, transpose_b=True),
                    shape=(
                        self.embedding_size,
                        -1,
                        self.layer_field_num[0] *
                        self.layer_field_num[layer_id]
                    )
                ),
                perm=[1, 0, 2]
            )
            # Feature Map
            state_z_k_conv = tf.nn.conv1d(
                input=state_z_k,
                filters=self.filter[layer_id],
                stride=1,
                padding='VALID'
            )
            state_z_k_conv_bias = tf.nn.bias_add(
                state_z_k_conv, self.bias[layer_id]
            )
            feature_map_k = tf.transpose(
                a=self.activation(state_z_k_conv_bias),
                perm=[0, 2, 1]
            )
            # Skip
            if not self.skip_connection:
                state_unit = feature_map_k
                output_unit = feature_map_k
            else:
                if layer_id != len(self.units) - 1:
                    state_unit, output_unit = tf.split(
                        feature_map_k,
                        [num_units // 2] * 2,
                        axis=1
                    )
                else:
                    state_unit = 0
                    output_unit = feature_map_k
            # Add
            cin_state_list.append(state_unit)
            output_list.append(output_unit)

        # Return
        outputs = tf.math.reduce_sum(
            input_tensor=tf.concat(output_list, axis=1),
            axis=-1,
            keepdims=False
        )
        return outputs


class KernelProduct(tf.keras.layers.Layer):

    def __init__(
        self,
        kernel_type,
        trainable=True,
        **kwargs
    ):
        super(KernelProduct, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.trainable = trainable

    def build(
        self,
        input_shape
    ):
        # Prep
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[2]
        num_interactions = int(self.field_size * (self.field_size - 1) // 2)
        # Kernel
        if self.kernel_type == 'mat':
            self.kernel = tf.Variable(
                tf.tile(
                    tf.expand_dims(
                        tf.eye(self.embedding_size, dtype=tf.float32), axis=1
                    ),
                    multiples=(1, num_interactions, 1)
                ),
                trainable=self.trainable,
                dtype=tf.float32
            )
        elif self.kernel_type == 'vec':
            self.kernel = tf.Variable(
                tf.ones(
                    shape=(num_interactions, self.embedding_size),
                    dtype=tf.float32
                ),
                trainable=self.trainable,
                dtype=tf.float32
            )
        elif self.kernel_type == 'num':
            self.kernel = tf.Variable(
                tf.ones(shape=(num_interactions, 1), dtype=tf.float32),
                trainable=self.trainable,
                dtype=tf.float32
            )

    def pairwise_feature(self, inputs, field_size):
        index_i = []
        index_j = []
        for i in range(0, field_size):
            for j in range(i+1, field_size):
                index_i.append(i)
                index_j.append(j)

        feature_i = tf.gather(inputs, index_i, axis=1)
        feature_j = tf.gather(inputs, index_j, axis=1)
        return feature_i, feature_j

    def call(
        self,
        inputs,
        training
    ):
        # Input
        feature_i, feature_j = self.pairwise_feature(
            inputs, self.field_size
        )
        # Product
        if self.kernel_type == 'mat':
            feature_ik = tf.math.reduce_sum(
                input_tensor=tf.multiply(
                    tf.expand_dims(feature_i, axis=1),
                    self.kernel
                ),
                axis=1
            )
        else:
            feature_ik = tf.multiply(feature_i, self.kernel)
        # Kernel Product
        kernel_product = tf.math.reduce_sum(
            input_tensor=tf.multiply(feature_ik, feature_j),
            axis=-1
        )

        return kernel_product


class LogarithmicNetwork(tf.keras.layers.Layer):

    def __init__(
        self,
        units,
        **kwargs
    ):
        super(LogarithmicNetwork, self).__init__(**kwargs)
        self.units = units

        self.dense_layer = EmbedDense(
            units=self.units,
            activation=tf.keras.activations.linear,
            use_bias=False
        )
        self.log_batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.exp_batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.flatten_layer = tf.keras.layers.Flatten()

    def call(self, inputs, training=False):
        inputs = tf.math.abs(inputs)
        inputs = tf.clip_by_value(
            inputs, 1e-4, 1e+4
        )
        log_inputs = tf.math.log(inputs)
        log_inputs = self.log_batch_norm_layer(log_inputs, training)
        log_outputs = self.dense_layer(log_inputs)
        exp_outputs = tf.math.exp(log_outputs)
        exp_outputs = self.exp_batch_norm_layer(exp_outputs, training)
        outputs = self.flatten_layer(exp_outputs)
        return outputs


class CNN1D(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        filters,
        kernel_size,
        pool_size,
        activation=None,
        **kwargs
    ):
        super(CNN1D, self).__init__(**kwargs)
        self.units = units
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation

        self.dnn = tf.keras.Sequential([
            EmbedDense(
                units=self.units,
                activation=self.activation
            ),
            tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation=self.activation
            ),
            tf.keras.layers.MaxPool1D(pool_size=self.pool_size)
        ])

    def call(self, inputs, training=False):
        outputs = self.dnn(inputs)
        return outputs


class DNN(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        output_units,
        activation=None,
        dropout=None,
        batch_norm=None,
        kernel_initializer=None,
        kernel_regularizer=None,
        **kwargs
    ):
        # Init
        super(DNN, self).__init__(**kwargs)
        self.dnn = tf.keras.Sequential()

        # Input Layer
        if dropout is not None:
            self.dnn.add(
                tf.keras.layers.Dropout(rate=dropout[0])
            )

        # Hidden Layer
        for i, num_neuron in enumerate(hidden_units):

            self.dnn.add(
                tf.keras.layers.Dense(
                    units=num_neuron,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )

            if batch_norm is not None and batch_norm[i]:
                self.dnn.add(
                    tf.keras.layers.BatchNormalization()
                )

            if dropout is not None and dropout[i+1] > 0:
                self.dnn.add(
                    tf.keras.layers.Dropout(rate=dropout[i+1])
                )

        # Output Layer
        self.dnn.add(
            tf.keras.layers.Dense(
                units=output_units,
                activation=tf.keras.activations.linear,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        )

    def call(self, inputs, training=False):
        outputs = self.dnn(
            inputs=inputs,
            training=training
        )
        return outputs


class ResidualDNN(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        output_units,
        activation=None,
        dropout=None,
        kernel_initializer=None,
        kernel_regularizer=None,
        **kwargs
    ):
        super(ResidualDNN, self).__init__(**kwargs)
        self.residual_dnn = tf.keras.Sequential()

        # Hidden Layer
        for i, num_neuron in enumerate(hidden_units):

            self.residual_dnn.add(
                ResidualDense(
                    units=num_neuron,
                    activation=activation,
                    dropout=dropout[i] if dropout is not None else None,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )

        # Output Layer
        self.residual_dnn.add(
            tf.keras.layers.Dense(
                units=output_units,
                activation=tf.keras.activations.linear,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        )

    def call(self, inputs, training=False):
        outputs = self.residual_dnn(
            inputs=inputs,
            training=training
        )
        return outputs


class GatedDNN(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        output_units,
        reduction_ratio,
        activation=None,
        dropout=None,
        kernel_initializer=None,
        kernel_regularizer=None,
        **kwargs
    ):
        super(GatedDNN, self).__init__(**kwargs)
        self.num_layers = len(hidden_units)
        self.dropout_layers = []
        self.gated_dense_layers = []

        # Dropout Layer
        if dropout is not None:
            for i in range(self.num_layers + 1):
                if dropout[i] > 0:
                    self.dropout_layers.append(
                        tf.keras.layers.Dropout(rate=dropout[i])
                    )
                else:
                    self.dropout_layers.append(None)

        # Hidden Layer
        for i, num_neuron in enumerate(hidden_units):

            self.gated_dense_layers.append(
                GatedDense(
                    units=num_neuron,
                    reduction_ratio=reduction_ratio,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )

        # Output_Layer
        self.output_layer = tf.keras.layers.Dense(
            units=output_units,
            activation=tf.keras.activations.linear,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs, training=False):
        logits = inputs

        if self.dropout_layers and self.dropout_layers[0]:
            logits = self.dropout_layers[0](logits, training)

        for i in range(self.num_layers):
            logits = self.gated_dense_layers[i](
                (logits, tf.stop_gradient(inputs)),
                training
            )
            if self.dropout_layers and self.dropout_layers[i+1]:
                logits = self.dropout_layers[i+1](logits, training)

        outputs = self.output_layer(
            logits,
            training=training
        )
        return outputs


class Sequential(tf.keras.layers.Layer):

    def __init__(
        self,
        layers,
        **kwargs
    ):
        super(Sequential, self).__init__(**kwargs)
        self.layers = layers

    def call(self, inputs, training=False):
        outputs = inputs

        for layer in self.layers:
            if 'training' in layer._call_full_argspec.args:
                outputs = layer(outputs, training=training)
            else:
                outputs = layer(outputs)

        return outputs


class Expert(tf.keras.layers.Layer):

    def __init__(
        self,
        layer,
        require_input_feature=False,
        **kwargs
    ):
        super(Expert, self).__init__(**kwargs)
        self.layer = layer
        self.require_input_feature = require_input_feature

    def call(self, inputs, training=False):
        if 'training' in self.layer._call_full_argspec.args:
            outputs = self.layer(inputs, training)
        else:
            outputs = self.layer(inputs)
        return outputs


class SparseMoE(tf.keras.layers.Layer):

    def __init__(
        self,
        gating_layer,
        expert_layers,
        **kwargs
    ):
        super(SparseMoE, self).__init__(**kwargs)
        self.gating_layer = gating_layer
        self.expert_layers = expert_layers

    def call(self, inputs, training=False):
        xl, x0 = inputs

        gates = self.gating_layer(
            inputs=tf.stop_gradient(x0),
            training=training
        )

        dispatcher = SparseDispatcher2D(gates)

        dispatched_xl = dispatcher.dispatch(xl)
        dispatched_x0 = dispatcher.dispatch(x0)

        dispatched_xl_1 = []
        for i, expert_layer in enumerate(self.expert_layers):
            expert_input_xl = dispatched_xl[i]
            expert_input_x0 = dispatched_x0[i]

            if expert_layer.require_input_feature:
                expert_output = expert_layer(
                    [expert_input_xl, expert_input_x0],
                    training=training
                )
            else:
                expert_output = expert_layer(
                    expert_input_xl,
                    training=training
                )

            dispatched_xl_1.append(expert_output)

        xl_1 = dispatcher.combine(dispatched_xl_1)

        return xl_1


class SparseFusion(tf.keras.layers.Layer):

    def __init__(
        self,
        mode,
        activation,
        **kwargs
    ):
        super(SparseFusion, self).__init__(**kwargs)
        self.mode = mode
        self.activation = activation
        assert self.mode in ('vector', 'bit')
        assert self.activation in (
            'sigmoid', 'smoothstep',
            'softmax', 'sparsemax'
        )

        if self.activation == 'sigmoid':
            self.activation_layer = tf.keras.layers.Lambda(
                function=lambda x: tf.nn.sigmoid(x) * 2.0
            )
        elif self.activation == 'smoothstep':
            self.activation_layer = tf.keras.Sequential([
                SmoothStep(gamma=2.0),
                tf.keras.layers.Lambda(function=lambda x: x * 2.0)
            ])
        elif self.activation == 'softmax':
            self.activation_layer = tf.keras.layers.Softmax(axis=0)
        elif self.activation == 'sparsemax':
            self.activation_layer = tf.keras.layers.Lambda(
                function=lambda x: tfa.activations.sparsemax(x, axis=0)
            )

    def build(self, input_shape):
        feature_dim = input_shape[1]
        fusion_dim = input_shape[-1]

        if self.mode == 'vector':
            self.logits = tf.Variable(
                initial_value=tf.zeros(shape=(fusion_dim, )),
                trainable=True,
                dtype=tf.float32
            )
            if self.activation in ('softmax', 'sparsemax'):
                self.temperature = tf.Variable(
                    initial_value=tf.ones(shape=(1, )),
                    trainable=True,
                    dtype=tf.float32
                )
        elif self.mode == 'bit':
            self.logits = tf.Variable(
                initial_value=tf.zeros(shape=(fusion_dim, feature_dim)),
                trainable=True,
                dtype=tf.float32
            )
            if self.activation in ('softmax', 'sparsemax'):
                self.temperature = tf.Variable(
                    initial_value=tf.ones(shape=(1, feature_dim)),
                    trainable=True,
                    dtype=tf.float32
                )

    def call(self, inputs, training=False):
        if self.activation in ('sigmoid', 'smoothstep'):
            gating_logits = self.activation_layer(
                self.logits
            )
        elif self.activation in ('softmax', 'sparsemax'):
            gating_logits = self.activation_layer(
                self.logits / self.temperature
            )

        if self.mode == 'vector':
            outputs = tf.einsum(
                'bf...n, n -> bf...',
                inputs,
                gating_logits
            )
        elif self.mode == 'bit':
            outputs = tf.einsum(
                'bf...n, nf -> bf...',
                inputs,
                gating_logits
            )

        return outputs


class SequentialMoE(tf.keras.layers.Layer):

    def __init__(
        self,
        layers,
        skip_connection=False,
        layer_norm=False,
        **kwargs
    ):
        super(SequentialMoE, self).__init__(**kwargs)
        self.layers = layers
        self.skip_connection = skip_connection
        self.layer_norm = layer_norm

        if self.layer_norm:
            num_layers = len(layers)
            self.normalization_layers = [
                tfa.layers.InstanceNormalization()
                for _ in range(num_layers)
            ]

    def call(self, inputs, training=False):
        outputs = inputs

        for i, layer in enumerate(self.layers):

            if self.skip_connection:
                outputs += layer([outputs, inputs], training=training)
            else:
                outputs = layer([outputs, inputs], training=training)

            if self.layer_norm:
                outputs = self.normalization_layers[i](
                    outputs, training=training
                )

        return outputs


class SequentialFusion(tf.keras.layers.Layer):

    def __init__(
        self,
        layers_list,
        mode,
        activation,
        skip_connection=False,
        layer_norm=False,
        **kwargs
    ):
        super(SequentialFusion, self).__init__(**kwargs)
        self.layers_list = layers_list
        self.skip_connection = skip_connection
        self.layer_norm = layer_norm

        self.fusion_layers_list = []
        for i, layers in enumerate(layers_list):
            if i == 0:
                continue
            else:
                fusion_layers = [
                    SparseFusion(mode=mode, activation=activation)
                    for _ in range(len(layers))
                ]
            self.fusion_layers_list.append(fusion_layers)

        self.output_fusion_layer = SparseFusion(
            mode=mode, activation=activation
        )

        if self.layer_norm:
            self.normalization_layers = [
                tfa.layers.InstanceNormalization()
                for _ in range(len(self.layers_list))
            ]

    def call(self, inputs, training=False):

        for i, layers in enumerate(self.layers_list):

            layers_output_list = []
            for j, layer in enumerate(layers):
                if i == 0:
                    layer_output = layer(
                        [inputs, inputs],
                        training=training
                    )
                    if self.skip_connection:
                        layer_output += inputs
                else:
                    layers_output_fused = self.fusion_layers_list[i-1][j](
                        layers_output_stacked
                    )
                    layer_output = layer(
                        [layers_output_fused, inputs],
                        training=training
                    )
                    if self.skip_connection:
                        layer_output += layers_output_fused
                layers_output_list.append(layer_output)

            layers_output_stacked = tf.stack(layers_output_list, axis=-1)

            if self.layer_norm:
                layers_output_stacked = self.normalization_layers[i](
                    layers_output_stacked, training=training
                )

        outputs = self.output_fusion_layer(layers_output_stacked)
        return outputs


class Identity(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        return inputs


class SequentialDynamicMoE(tf.keras.layers.Layer):

    def __init__(
        self,
        layers,
        estimators,
        router,
        skip_connection=False,
        layer_norm=False,
        **kwargs
    ):
        super(SequentialDynamicMoE, self).__init__(**kwargs)
        self.layers = layers
        self.estimators = estimators
        self.router = router
        self.skip_connection = skip_connection
        self.layer_norm = layer_norm

        self.num_layers = len(layers)

        if self.layer_norm:
            self.normalization_layers = [
                tfa.layers.InstanceNormalization()
                for _ in range(self.num_layers)
            ]

    def compute_prediction(self, inputs, depth, training=False):

        last_layer_output, init_input, gates = inputs
        current_layer_output = self.layers[depth](
            [last_layer_output, init_input],
            training=training
        )
        last_layer_output_shape = [None] + last_layer_output.shape.as_list()[1:]
        init_input_shape = [None] + init_input.shape.as_list()[1:]
        gates_shape = [None] + gates.shape.as_list()[1:]

        if self.skip_connection:
            current_layer_output += last_layer_output

        if self.layer_norm:
            current_layer_output = self.normalization_layers[i](
                current_layer_output, training=training
            )

        if depth == self.num_layers - 1:
            preds = self.estimators[depth](
                current_layer_output,
                training=training
            )
        else:
            current_layer_gates_raw = tf.stack(
                [
                    gates[:, depth],
                    tf.math.reduce_sum(gates[:, (depth + 1):], axis=1)
                ],
                axis=1
            )
            current_layer_gates_normalized = tf.linalg.normalize(
                current_layer_gates_raw, ord=1, axis=1
            )[0]
            dispatcher = SparseDispatcher2D(current_layer_gates_normalized)

            current_layer_output_exit, current_layer_output_enter = \
                dispatcher.dispatch(current_layer_output)
            _, init_input_enter = dispatcher.dispatch(init_input)
            _, gates_enter = dispatcher.dispatch(gates)

            current_layer_output_exit = tf.ensure_shape(
                current_layer_output_exit,
                shape=last_layer_output_shape
            )
            current_layer_output_enter = tf.ensure_shape(
                current_layer_output_enter,
                shape=last_layer_output_shape
            )
            init_input_enter = tf.ensure_shape(
                init_input_enter,
                shape=init_input_shape
            )
            gates_enter = tf.ensure_shape(
                gates_enter,
                shape=gates_shape
            )

            current_layer_preds_exit = self.estimators[depth](
                current_layer_output_exit,
                training=training
            )
            next_layer_preds_enter = self.compute_prediction(
                inputs=[
                    current_layer_output_enter,
                    init_input_enter,
                    gates_enter
                ],
                depth=depth + 1,
                training=training
            )
            preds = dispatcher.combine(
                [current_layer_preds_exit, next_layer_preds_enter]
            )

        return preds

    def call(self, inputs, training=False):
        gates = self.router(tf.stop_gradient(inputs))
        outputs = self.compute_prediction(
            [inputs, inputs, gates],
            depth=0,
            training=training
        )
        return outputs


class RankingModel(object):

    def __init__(
        self,
        input_layer,
        interaction_layer,
        output_layer
    ):
        self.input_layer = input_layer
        self.interaction_layer = interaction_layer
        self.output_layer = output_layer

        self.model = tf.keras.Sequential(
            [self.interaction_layer, self.output_layer]
        )

    def call(self, inputs, training=False):
        embeddings = self.input_layer(inputs)
        preds = self.model(embeddings, training)
        return preds

    @property
    def embedding_trainable_variables(self):
        trainable_variables = self.input_layer.trainable_variables
        return trainable_variables

    @property
    def model_trainable_variables(self):
        trainable_variables = self.interaction_layer.trainable_variables + \
            self.output_layer.trainable_variables
        return trainable_variables

    def compile(
        self,
        embedding_optimizer,
        model_optimizer,
        loss_fn,
        metrics,
        jit_compile
    ):
        self.embedding_optimizer = embedding_optimizer
        self.model_optimizer = model_optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.jit_compile = jit_compile

    def _reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def _update_metrics(self, label, pred):
        for metric in self.metrics:
            metric.update_state(label, pred)

    def _compute_metrics(self):
        logs = {
            metric.name: metric.result()
            for metric in self.metrics
        }
        return logs

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            preds = self.call(x, training=True)
            loss = self.loss_fn(y, preds)
            regularizations = self.input_layer.losses + \
                self.interaction_layer.losses + \
                self.output_layer.losses

            if regularizations:
                objective = loss + tf.math.add_n(regularizations)
            else:
                objective = loss

        embedding_grads = tape.gradient(
            target=objective,
            sources=self.embedding_trainable_variables
        )
        self.embedding_optimizer.apply_gradients(
            grads_and_vars=list(zip(
                embedding_grads,
                self.embedding_trainable_variables
            ))
        )

        model_grads = tape.gradient(
            target=objective,
            sources=self.model_trainable_variables
        )
        self.model_optimizer.apply_gradients(
            grads_and_vars=list(zip(
                model_grads,
                self.model_trainable_variables
            ))
        )

        del tape

        self._update_metrics(y, preds)

        return None

    def test_step(self, data):
        x, y = data
        preds = self.call(x, training=False)

        self._update_metrics(y, preds)

        return None

    def train(self, iterator, metric_freq):
        self._reset_metrics()
        callbacks = tf.keras.callbacks.CallbackList(
            callbacks=None,
            add_history=True,
            add_progbar=True,
        )

        if self.jit_compile:
            train_step = tf.function(self.train_step)
        else:
            train_step = self.train_step

        callbacks.on_train_begin()
        for step, data in enumerate(iterator):
            callbacks.on_train_batch_begin(step)
            train_step(data)
            callbacks.on_train_batch_end(step)

            if step % metric_freq == 0:
                callbacks.on_train_batch_end(
                    step,
                    logs=self._compute_metrics()
                )
            else:
                callbacks.on_train_batch_end(step)
        callbacks.on_train_end()

        return None

    def test(self, iterator):
        self._reset_metrics()
        callbacks = tf.keras.callbacks.CallbackList(
            callbacks=None,
            add_history=True,
            add_progbar=True,
        )

        if self.jit_compile:
            test_step = tf.function(self.test_step)
        else:
            test_step = self.test_step

        callbacks.on_test_begin()
        for step, data in enumerate(iterator):
            callbacks.on_test_batch_begin(step)
            test_step(data)
            callbacks.on_test_batch_end(step)
        callbacks.on_test_end(logs=self._compute_metrics())

        return None


class GFTRL(tf.keras.optimizers.Optimizer):

    def __init__(
        self,
        alpha=0.01,
        beta=1e-4,
        lambda1=0.0,
        lambda2=0.0,
        name='GFTRL'
    ):
        super(GFTRL, self).__init__(name)
        self._set_hyper('alpha', alpha)
        self._set_hyper('beta', beta)
        self._set_hyper('lambda1', lambda1)
        self._set_hyper('lambda2', lambda2)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'n')
        for var in var_list:
            self.add_slot(var, 'z')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(GFTRL, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(dict(
                alpha=tf.identity(self._get_hyper('alpha', var_dtype)),
                beta=tf.identity(self._get_hyper('beta', var_dtype)),
                lambda1=tf.identity(self._get_hyper('lambda1', var_dtype)),
                lambda2=tf.identity(self._get_hyper('lambda2', var_dtype))
        ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (
            (apply_state or {}).get((var_device, var_dtype)) or
            self._fallback_apply_state(var_device, var_dtype)
        )

        # Init
        n = self.get_slot(var, 'n')
        z = self.get_slot(var, 'z')
        input_dim = tf.convert_to_tensor(var.shape[0], dtype=var_dtype)
        output_dim = tf.convert_to_tensor(var.shape[1], dtype=var_dtype)
        # Compute
        grad2 = tf.square(grad)
        sigma = (tf.sqrt(n + grad2) - tf.sqrt(n)) / coefficients['alpha']
        new_n = n.assign(n + grad2)
        new_z = z.assign(z + grad - sigma * var)
        z_norm = tf.norm(new_z, ord='euclidean', axis=1, keepdims=True)
        # Update
        new_var = var.assign(
            tf.where(
                condition=tf.math.less_equal(
                    z_norm,
                    coefficients['lambda1'] * tf.sqrt(output_dim)
                ),
                x=tf.zeros(shape=(input_dim, output_dim)),
                y=new_z * tf.divide(
                    (coefficients['lambda1'] * tf.sqrt(output_dim)) /
                    z_norm - 1.0,
                    (coefficients['beta'] + tf.sqrt(new_n)) /
                    coefficients['alpha'] + coefficients['lambda2']
                )
            )
        )
        # Return
        updates = [new_var, new_n, new_z]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._resource_apply_dense(
            tf.convert_to_tensor(
                tf.IndexedSlices(grad, indices, tf.shape(handle))
            ),
            handle
        )


class GRDA(tf.keras.optimizers.Optimizer):

    def __init__(
        self,
        learning_rate=0.005,
        c=0.005,
        mu=0.7,
        name='GRDA'
    ):
        super(GRDA, self).__init__(name)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('c', c)
        self._set_hyper('mu', mu)
        self._set_hyper('iter', tf.Variable(0.0))
        self._set_hyper('l1_accumulator', tf.Variable(0.0))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'accumulator')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(GRDA, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(dict(
                learning_rate=tf.identity(
                    self._get_hyper('learning_rate', var_dtype)
                ),
                c=tf.identity(
                    self._get_hyper('c', var_dtype)
                ),
                mu=tf.identity(
                    self._get_hyper('mu', var_dtype)
                ),
                iter=self._get_hyper('iter', var_dtype),
                l1_accumulator=self._get_hyper('l1_accumulator', var_dtype)
        ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (
            (apply_state or {}).get((var_device, var_dtype)) or
            self._fallback_apply_state(var_device, var_dtype)
        )

        # Init
        iter = coefficients['iter']
        l1_accumulator = coefficients['l1_accumulator']
        accumulator = self.get_slot(var, 'accumulator')
        # Compute
        l1_diff = coefficients['c'] *\
            tf.math.pow(
                coefficients['learning_rate'],
                coefficients['mu'] + 0.5
            ) * (
                tf.math.pow(iter + 1.0, coefficients['mu']) -
                tf.math.pow(iter, coefficients['mu'])
            )
        # Update
        new_iter = iter.assign(iter + 1.0)
        new_l1_accumulator = l1_accumulator.assign(l1_accumulator + l1_diff)
        new_accumulator = accumulator.assign(
            accumulator +
            tf.math.maximum(1.0 - iter, 0.0) * var -
            coefficients['learning_rate'] * grad
        )
        new_var = var.assign(
            tf.math.sign(accumulator) *
            tf.math.maximum(tf.math.abs(accumulator) - l1_accumulator, 0.0)
        )

        # Return
        updates = [new_iter, new_l1_accumulator, new_accumulator, new_var]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._resource_apply_dense(
            tf.convert_to_tensor(
                tf.IndexedSlices(grad, indices, tf.shape(handle))
            ),
            handle
        )
