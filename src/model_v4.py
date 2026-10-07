"""Conditional onset-group LSTM model for NeuraTune V4."""

import tensorflow as tf
from tensorflow import keras

PITCH_EMBEDDING_DIMENSION = 32
DURATION_EMBEDDING_DIMENSION = 16
DELTA_EMBEDDING_DIMENSION = 16
LSTM_UNITS = 256
DECODER_UNITS = 256
DROPOUT_RATE = 0.3
PAD_CLASS_INDEX = 0


@keras.utils.register_keras_serializable(package="neuratune")
class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Cross-entropy over non-PAD target positions."""

    def __init__(self, pad_class_index: int = PAD_CLASS_INDEX, name: str = "masked_sparse_cce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_class_index = pad_class_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        losses = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        valid = tf.cast(tf.not_equal(y_true, self.pad_class_index), losses.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(losses * valid), tf.reduce_sum(valid))

    def get_config(self):
        return {**super().get_config(), "pad_class_index": self.pad_class_index}


@keras.utils.register_keras_serializable(package="neuratune")
class WeightedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Moderately class-weighted loss, optionally masking PAD targets."""

    def __init__(self, class_weights, pad_class_index=None, name="weighted_sparse_cce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = [float(value) for value in class_weights]
        self.pad_class_index = pad_class_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        losses = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weights = tf.gather(tf.constant(self.class_weights, dtype=losses.dtype), y_true)
        valid = tf.ones_like(losses, dtype=losses.dtype)
        if self.pad_class_index is not None:
            valid = tf.cast(tf.not_equal(y_true, self.pad_class_index), losses.dtype)
        weights *= valid
        return tf.math.divide_no_nan(tf.reduce_sum(losses * weights), tf.reduce_sum(weights))

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights, "pad_class_index": self.pad_class_index}


@keras.utils.register_keras_serializable(package="neuratune")
class MaskedSparseCategoricalAccuracy(keras.metrics.Metric):
    """Accuracy over target classes other than PAD."""

    def __init__(self, pad_class_index: int = PAD_CLASS_INDEX, name: str = "non_pad_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_class_index = pad_class_index
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.not_equal(y_true, self.pad_class_index)
        correct = tf.logical_and(valid, tf.equal(y_true, predicted))
        valid_values = tf.cast(valid, self.dtype)
        correct_values = tf.cast(correct, self.dtype)
        if sample_weight is not None:
            weights = tf.cast(sample_weight, self.dtype)
            valid_values *= weights
            correct_values *= weights
        self.correct.assign_add(tf.reduce_sum(correct_values))
        self.count.assign_add(tf.reduce_sum(valid_values))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.count)

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "pad_class_index": self.pad_class_index}


@keras.utils.register_keras_serializable(package="neuratune")
class RealPitchAccuracy(MaskedSparseCategoricalAccuracy):
    """Accuracy on real pitches, excluding PAD and EOS."""

    def __init__(self, eos_class_index: int, name: str = "real_pitch_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eos_class_index = int(eos_class_index)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.logical_and(tf.not_equal(y_true, self.pad_class_index), tf.not_equal(y_true, self.eos_class_index))
        correct = tf.logical_and(valid, tf.equal(y_true, predicted))
        valid_values = tf.cast(valid, self.dtype)
        correct_values = tf.cast(correct, self.dtype)
        if sample_weight is not None:
            weights = tf.cast(sample_weight, self.dtype)
            valid_values *= weights
            correct_values *= weights
        self.correct.assign_add(tf.reduce_sum(correct_values))
        self.count.assign_add(tf.reduce_sum(valid_values))

    def get_config(self):
        return {**super().get_config(), "eos_class_index": self.eos_class_index}


@keras.utils.register_keras_serializable(package="neuratune")
class EOSAccuracy(keras.metrics.Metric):
    """Accuracy restricted to explicit EOS target positions."""

    def __init__(self, eos_class_index: int, name: str = "eos_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eos_class_index = int(eos_class_index)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.equal(y_true, self.eos_class_index)
        correct = tf.logical_and(valid, tf.equal(predicted, self.eos_class_index))
        weights = tf.cast(valid, self.dtype)
        correct_values = tf.cast(correct, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            weights *= sample_weight
            correct_values *= sample_weight
        self.correct.assign_add(tf.reduce_sum(correct_values))
        self.count.assign_add(tf.reduce_sum(weights))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.count)

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "eos_class_index": self.eos_class_index}


def build_v4_conditional_model(
    sequence_length: int,
    max_pitches_per_onset: int,
    pitch_class_count: int,
    duration_class_count: int,
    delta_class_count: int,
    eos_pitch_class_index: int,
    start_pitch_class_index: int,
    delta_class_weights=None,
    duration_class_weights=None,
) -> keras.Model:
    """Build the V4 encoder and EOS-aware teacher-forced decoder.

    ``pitch_class_count`` includes PAD and real pitch classes. EOS is an
    additional output class; START is used only by the decoder embedding.
    """
    pitch_input = keras.Input(shape=(sequence_length, max_pitches_per_onset), dtype="int32", name="pitch_input")
    duration_input = keras.Input(shape=(sequence_length, max_pitches_per_onset), dtype="int32", name="duration_input")
    delta_input = keras.Input(shape=(sequence_length,), dtype="int32", name="delta_input")
    target_pitch_input = keras.Input(shape=(max_pitches_per_onset,), dtype="int32", name="target_pitch_input")

    pitch_embedding = keras.layers.Embedding(pitch_class_count, PITCH_EMBEDDING_DIMENSION, name="pitch_embedding")(pitch_input)
    duration_embedding = keras.layers.Embedding(duration_class_count, DURATION_EMBEDDING_DIMENSION, name="duration_embedding")(duration_input)
    aligned_slots = keras.layers.Concatenate(axis=-1, name="aligned_pitch_duration_slots")([pitch_embedding, duration_embedding])
    flattened_slots = keras.layers.Reshape((sequence_length, max_pitches_per_onset * (PITCH_EMBEDDING_DIMENSION + DURATION_EMBEDDING_DIMENSION)), name="flattened_historical_onsets")(aligned_slots)
    delta_embedding = keras.layers.Embedding(delta_class_count, DELTA_EMBEDDING_DIMENSION, name="delta_embedding")(delta_input)
    historical_features = keras.layers.Concatenate(axis=-1, name="historical_onset_features")([flattened_slots, delta_embedding])
    context = keras.layers.LSTM(LSTM_UNITS, name="encoder_lstm")(historical_features)
    context = keras.layers.Dropout(DROPOUT_RATE, name="encoder_dropout")(context)

    decoder_vocab_size = start_pitch_class_index + 1
    decoder_embedding = keras.layers.Embedding(decoder_vocab_size, PITCH_EMBEDDING_DIMENSION, name="target_pitch_decoder_embedding")(target_pitch_input)
    decoder_states = keras.layers.GRU(DECODER_UNITS, return_sequences=True, name="conditional_pitch_decoder")(decoder_embedding, initial_state=context)
    pitch_output = keras.layers.TimeDistributed(keras.layers.Dense(eos_pitch_class_index + 1, activation="softmax"), name="pitch_output")(decoder_states)
    duration_features = keras.layers.Concatenate(axis=-1, name="pitch_conditioned_duration_features")([decoder_states, pitch_output])
    duration_output = keras.layers.TimeDistributed(keras.layers.Dense(duration_class_count, activation="softmax"), name="duration_output")(duration_features)
    delta_output = keras.layers.Dense(delta_class_count, activation="softmax", name="delta_output")(context)

    model = keras.Model(
        inputs={"pitch_input": pitch_input, "duration_input": duration_input, "delta_input": delta_input, "target_pitch_input": target_pitch_input},
        outputs={"pitch_output": pitch_output, "duration_output": duration_output, "delta_output": delta_output},
        name="neuratune_v4_conditional_onset_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={
            "pitch_output": MaskedSparseCategoricalCrossentropy(),
            # Production V4 uses the original unweighted rhythm objective.
            "duration_output": MaskedSparseCategoricalCrossentropy(pad_class_index=0),
            "delta_output": keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "pitch_output": [MaskedSparseCategoricalAccuracy(name="active_decoder_accuracy"), RealPitchAccuracy(eos_pitch_class_index, name="real_pitch_accuracy"), EOSAccuracy(eos_pitch_class_index)],
            "duration_output": [MaskedSparseCategoricalAccuracy(name="duration_accuracy")],
            "delta_output": [keras.metrics.SparseCategoricalAccuracy(name="delta_accuracy")],
        },
    )
    model.eos_pitch_class_index = int(eos_pitch_class_index)
    model.start_pitch_class_index = int(start_pitch_class_index)
    return model
