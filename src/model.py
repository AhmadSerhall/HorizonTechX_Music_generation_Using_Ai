"""TensorFlow/Keras model definition for next-event music prediction."""

from tensorflow import keras


EMBEDDING_DIMENSION = 64
LSTM_UNITS = 64
DROPOUT_RATE = 0.2


def build_lstm_model(vocabulary_size: int, sequence_length: int) -> keras.Model:
    """Build and compile a small LSTM that predicts the next event ID."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length,), dtype="int32"),
            keras.layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=EMBEDDING_DIMENSION,
                name="event_embedding",
            ),
            keras.layers.LSTM(LSTM_UNITS, name="lstm"),
            keras.layers.Dropout(DROPOUT_RATE, name="dropout"),
            keras.layers.Dense(vocabulary_size, activation="softmax", name="next_event"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
