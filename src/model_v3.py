"""Padded onset-group LSTM model for NeuraTune V3 preprocessing artifacts."""

import tensorflow as tf
from tensorflow import keras


PITCH_EMBEDDING_DIMENSION = 32
DURATION_EMBEDDING_DIMENSION = 16
DELTA_EMBEDDING_DIMENSION = 16
LSTM_UNITS = 256
DROPOUT_RATE = 0.3
PAD_CLASS_INDEX = 0


@keras.utils.register_keras_serializable(package="neuratune")
class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Sparse cross-entropy that excludes padded pitch or duration target slots."""

    def __init__(
        self,
        pad_class_index: int = PAD_CLASS_INDEX,
        name: str = "masked_sparse_cce",
        **kwargs,
    ):
        # ``reduction`` is supplied when Keras reloads a saved .keras model.
        super().__init__(name=name, **kwargs)
        self.pad_class_index = pad_class_index

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        per_slot_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        valid_slots = tf.cast(tf.not_equal(y_true, self.pad_class_index), per_slot_loss.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(per_slot_loss * valid_slots), tf.reduce_sum(valid_slots)
        )

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "pad_class_index": self.pad_class_index}


@keras.utils.register_keras_serializable(package="neuratune")
class MaskedSparseCategoricalAccuracy(keras.metrics.Metric):
    """Accuracy over valid non-padding target slots only."""

    def __init__(self, pad_class_index: int = PAD_CLASS_INDEX, name: str = "masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_class_index = pad_class_index
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None
    ) -> None:
        predicted_classes = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        valid_slots = tf.not_equal(y_true, self.pad_class_index)
        correct_slots = tf.logical_and(tf.equal(y_true, predicted_classes), valid_slots)
        valid_values = tf.cast(valid_slots, self.dtype)
        correct_values = tf.cast(correct_slots, self.dtype)
        if sample_weight is not None:
            weights = tf.cast(sample_weight, self.dtype)
            valid_values *= weights
            correct_values *= weights
        self.correct.assign_add(tf.reduce_sum(correct_values))
        self.count.assign_add(tf.reduce_sum(valid_values))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.correct, self.count)

    def reset_state(self) -> None:
        self.correct.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "pad_class_index": self.pad_class_index}


def build_v3_lstm_model(
    sequence_length: int,
    max_pitches_per_onset: int,
    pitch_class_count: int,
    duration_class_count: int,
    delta_class_count: int,
) -> keras.Model:
    """Build and compile the separate V3 onset-group LSTM model."""
    pitch_input = keras.Input(
        shape=(sequence_length, max_pitches_per_onset), dtype="int32", name="pitch_input"
    )
    duration_input = keras.Input(
        shape=(sequence_length, max_pitches_per_onset),
        dtype="int32",
        name="duration_input",
    )
    delta_input = keras.Input(shape=(sequence_length,), dtype="int32", name="delta_input")

    pitch_embedding = keras.layers.Embedding(
        input_dim=pitch_class_count,
        output_dim=PITCH_EMBEDDING_DIMENSION,
        name="pitch_embedding",
    )(pitch_input)
    duration_embedding = keras.layers.Embedding(
        input_dim=duration_class_count,
        output_dim=DURATION_EMBEDDING_DIMENSION,
        name="duration_embedding",
    )(duration_input)
    slot_features = keras.layers.Concatenate(axis=-1, name="aligned_slot_features")(
        [pitch_embedding, duration_embedding]
    )
    flattened_slots = keras.layers.Reshape(
        (sequence_length, max_pitches_per_onset * (PITCH_EMBEDDING_DIMENSION + DURATION_EMBEDDING_DIMENSION)),
        name="flattened_onset_slots",
    )(slot_features)
    delta_embedding = keras.layers.Embedding(
        input_dim=delta_class_count,
        output_dim=DELTA_EMBEDDING_DIMENSION,
        name="delta_embedding",
    )(delta_input)
    timestep_features = keras.layers.Concatenate(axis=-1, name="onset_group_features")(
        [flattened_slots, delta_embedding]
    )
    context = keras.layers.LSTM(LSTM_UNITS, name="lstm")(timestep_features)
    context = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(context)

    pitch_logits = keras.layers.Dense(
        max_pitches_per_onset * pitch_class_count, name="pitch_logits"
    )(context)
    pitch_output = keras.layers.Softmax(name="pitch_output")(
        keras.layers.Reshape(
            (max_pitches_per_onset, pitch_class_count), name="pitch_slot_logits"
        )(pitch_logits)
    )
    duration_logits = keras.layers.Dense(
        max_pitches_per_onset * duration_class_count, name="duration_logits"
    )(context)
    duration_output = keras.layers.Softmax(name="duration_output")(
        keras.layers.Reshape(
            (max_pitches_per_onset, duration_class_count), name="duration_slot_logits"
        )(duration_logits)
    )
    delta_output = keras.layers.Dense(
        delta_class_count, activation="softmax", name="delta_output"
    )(context)

    model = keras.Model(
        inputs={
            "pitch_input": pitch_input,
            "duration_input": duration_input,
            "delta_input": delta_input,
        },
        outputs={
            "pitch_output": pitch_output,
            "duration_output": duration_output,
            "delta_output": delta_output,
        },
        name="neuratune_v3_onset_group_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={
            "pitch_output": MaskedSparseCategoricalCrossentropy(),
            "duration_output": MaskedSparseCategoricalCrossentropy(),
            "delta_output": keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "pitch_output": [MaskedSparseCategoricalAccuracy(name="accuracy")],
            "duration_output": [MaskedSparseCategoricalAccuracy(name="accuracy")],
            "delta_output": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        },
    )
    return model
