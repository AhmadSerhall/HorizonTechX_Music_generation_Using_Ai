"""V4E conditional onset model with compact, unweighted rhythm heads."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

PITCH_EMBEDDING_DIMENSION = 32
DURATION_EMBEDDING_DIMENSION = 16
DELTA_EMBEDDING_DIMENSION = 16
LSTM_UNITS = 256
DECODER_UNITS = 256
DROPOUT_RATE = 0.3
PAD_CLASS_INDEX = 0


@keras.utils.register_keras_serializable(package="neuratune_v4e")
class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    def __init__(self, pad_class_index: int = 0, name: str = "v4e_masked_sparse_cce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_class_index = int(pad_class_index)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        valid = tf.cast(tf.not_equal(y_true, self.pad_class_index), loss.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(loss * valid), tf.reduce_sum(valid))

    def get_config(self):
        return {**super().get_config(), "pad_class_index": self.pad_class_index}


@keras.utils.register_keras_serializable(package="neuratune_v4e")
class MaskedSparseCategoricalAccuracy(keras.metrics.Metric):
    def __init__(self, pad_class_index: int = 0, name: str = "masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_class_index = int(pad_class_index)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.not_equal(y_true, self.pad_class_index)
        correct = tf.logical_and(valid, tf.equal(pred, y_true))
        weights = tf.cast(valid, self.dtype)
        values = tf.cast(correct, self.dtype)
        if sample_weight is not None:
            weights *= tf.cast(sample_weight, self.dtype)
            values *= tf.cast(sample_weight, self.dtype)
        self.correct.assign_add(tf.reduce_sum(values)); self.count.assign_add(tf.reduce_sum(weights))

    def result(self): return tf.math.divide_no_nan(self.correct, self.count)
    def reset_state(self): self.correct.assign(0.0); self.count.assign(0.0)
    def get_config(self): return {**super().get_config(), "pad_class_index": self.pad_class_index}


@keras.utils.register_keras_serializable(package="neuratune_v4e")
class RealPitchAccuracy(MaskedSparseCategoricalAccuracy):
    def __init__(self, eos_class_index: int, name: str = "real_pitch_accuracy", **kwargs):
        super().__init__(name=name, **kwargs); self.eos_class_index = int(eos_class_index)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32); pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.logical_and(tf.not_equal(y_true, self.pad_class_index), tf.not_equal(y_true, self.eos_class_index))
        weights = tf.cast(valid, self.dtype); values = tf.cast(tf.logical_and(valid, tf.equal(pred, y_true)), self.dtype)
        if sample_weight is not None:
            weights *= tf.cast(sample_weight, self.dtype); values *= tf.cast(sample_weight, self.dtype)
        self.correct.assign_add(tf.reduce_sum(values)); self.count.assign_add(tf.reduce_sum(weights))

    def get_config(self): return {**super().get_config(), "eos_class_index": self.eos_class_index}


@keras.utils.register_keras_serializable(package="neuratune_v4e")
class EOSAccuracy(keras.metrics.Metric):
    def __init__(self, eos_class_index: int, name: str = "eos_accuracy", **kwargs):
        super().__init__(name=name, **kwargs); self.eos_class_index = int(eos_class_index)
        self.correct = self.add_weight(name="correct", initializer="zeros"); self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32); pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.equal(y_true, self.eos_class_index); weights = tf.cast(valid, self.dtype)
        values = tf.cast(tf.logical_and(valid, tf.equal(pred, self.eos_class_index)), self.dtype)
        if sample_weight is not None:
            weights *= tf.cast(sample_weight, self.dtype); values *= tf.cast(sample_weight, self.dtype)
        self.correct.assign_add(tf.reduce_sum(values)); self.count.assign_add(tf.reduce_sum(weights))

    def result(self): return tf.math.divide_no_nan(self.correct, self.count)
    def reset_state(self): self.correct.assign(0.0); self.count.assign(0.0)
    def get_config(self): return {**super().get_config(), "eos_class_index": self.eos_class_index}


def build_v4e_conditional_model(sequence_length: int, max_pitches_per_onset: int, pitch_class_count: int,
                                duration_class_count: int, delta_class_count: int,
                                eos_pitch_class_index: int, start_pitch_class_index: int) -> keras.Model:
    pitch_input = keras.Input((sequence_length, max_pitches_per_onset), dtype="int32", name="pitch_input")
    duration_input = keras.Input((sequence_length, max_pitches_per_onset), dtype="int32", name="duration_input")
    delta_input = keras.Input((sequence_length,), dtype="int32", name="delta_input")
    target_pitch_input = keras.Input((max_pitches_per_onset,), dtype="int32", name="target_pitch_input")
    p = keras.layers.Embedding(pitch_class_count, PITCH_EMBEDDING_DIMENSION, name="pitch_embedding")(pitch_input)
    d = keras.layers.Embedding(duration_class_count, DURATION_EMBEDDING_DIMENSION, name="duration_embedding")(duration_input)
    slots = keras.layers.Concatenate(axis=-1, name="aligned_pitch_duration_slots")([p, d])
    slots = keras.layers.Reshape((sequence_length, max_pitches_per_onset * (PITCH_EMBEDDING_DIMENSION + DURATION_EMBEDDING_DIMENSION)), name="flattened_historical_onsets")(slots)
    de = keras.layers.Embedding(delta_class_count, DELTA_EMBEDDING_DIMENSION, name="delta_embedding")(delta_input)
    features = keras.layers.Concatenate(axis=-1, name="historical_onset_features")([slots, de])
    context = keras.layers.Dropout(DROPOUT_RATE, name="encoder_dropout")(keras.layers.LSTM(LSTM_UNITS, name="encoder_lstm")(features))
    vocab = start_pitch_class_index + 1
    dec_in = keras.layers.Embedding(vocab, PITCH_EMBEDDING_DIMENSION, name="target_pitch_decoder_embedding")(target_pitch_input)
    dec = keras.layers.GRU(DECODER_UNITS, return_sequences=True, name="conditional_pitch_decoder")(dec_in, initial_state=context)
    pitch_out = keras.layers.TimeDistributed(keras.layers.Dense(eos_pitch_class_index + 1, activation="softmax"), name="pitch_output")(dec)
    duration_features = keras.layers.Concatenate(axis=-1, name="pitch_conditioned_duration_features")([dec, pitch_out])
    duration_out = keras.layers.TimeDistributed(keras.layers.Dense(duration_class_count, activation="softmax"), name="duration_output")(duration_features)
    delta_out = keras.layers.Dense(delta_class_count, activation="softmax", name="delta_output")(context)
    model = keras.Model({"pitch_input": pitch_input, "duration_input": duration_input, "delta_input": delta_input, "target_pitch_input": target_pitch_input}, {"pitch_output": pitch_out, "duration_output": duration_out, "delta_output": delta_out}, name="neuratune_v4e_compact_rhythm")
    model.compile(optimizer=keras.optimizers.Adam(), loss={"pitch_output": MaskedSparseCategoricalCrossentropy(), "duration_output": MaskedSparseCategoricalCrossentropy(), "delta_output": keras.losses.SparseCategoricalCrossentropy()}, metrics={"pitch_output": [MaskedSparseCategoricalAccuracy(name="active_decoder_accuracy"), RealPitchAccuracy(eos_pitch_class_index)], "duration_output": [MaskedSparseCategoricalAccuracy(name="duration_accuracy")], "delta_output": [keras.metrics.SparseCategoricalAccuracy(name="delta_accuracy")]})
    model.eos_pitch_class_index = int(eos_pitch_class_index); model.start_pitch_class_index = int(start_pitch_class_index)
    return model
