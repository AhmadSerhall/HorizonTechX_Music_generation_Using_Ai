"""Multi-head Keras model for NeuraTune's timing-aware V2 representation."""

from tensorflow import keras


PITCH_EMBEDDING_DIMENSION = 32
DELTA_EMBEDDING_DIMENSION = 16
DURATION_EMBEDDING_DIMENSION = 16
LSTM_UNITS = 256
DROPOUT_RATE = 0.3


def build_v2_lstm_model(
    sequence_length: int,
    pitch_class_count: int,
    delta_class_count: int,
    duration_class_count: int,
) -> keras.Model:
    """Build a three-input, three-output LSTM for V2 event attributes."""
    pitch_input = keras.Input(
        shape=(sequence_length,), dtype="int32", name="pitch_input"
    )
    delta_input = keras.Input(
        shape=(sequence_length,), dtype="int32", name="delta_input"
    )
    duration_input = keras.Input(
        shape=(sequence_length,), dtype="int32", name="duration_input"
    )

    pitch_embedding = keras.layers.Embedding(
        input_dim=pitch_class_count,
        output_dim=PITCH_EMBEDDING_DIMENSION,
        name="pitch_embedding",
    )(pitch_input)
    delta_embedding = keras.layers.Embedding(
        input_dim=delta_class_count,
        output_dim=DELTA_EMBEDDING_DIMENSION,
        name="delta_embedding",
    )(delta_input)
    duration_embedding = keras.layers.Embedding(
        input_dim=duration_class_count,
        output_dim=DURATION_EMBEDDING_DIMENSION,
        name="duration_embedding",
    )(duration_input)

    combined_features = keras.layers.Concatenate(name="attribute_embeddings")(
        [pitch_embedding, delta_embedding, duration_embedding]
    )
    context = keras.layers.LSTM(LSTM_UNITS, name="lstm")(combined_features)
    context = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(context)

    outputs = {
        "pitch_output": keras.layers.Dense(
            pitch_class_count, activation="softmax", name="pitch_output"
        )(context),
        "delta_output": keras.layers.Dense(
            delta_class_count, activation="softmax", name="delta_output"
        )(context),
        "duration_output": keras.layers.Dense(
            duration_class_count, activation="softmax", name="duration_output"
        )(context),
    }
    model = keras.Model(
        inputs={
            "pitch_input": pitch_input,
            "delta_input": delta_input,
            "duration_input": duration_input,
        },
        outputs=outputs,
        name="neuratune_v2_multihead_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={
            "pitch_output": keras.losses.SparseCategoricalCrossentropy(),
            "delta_output": keras.losses.SparseCategoricalCrossentropy(),
            "duration_output": keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "pitch_output": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "delta_output": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "duration_output": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        },
    )
    return model
