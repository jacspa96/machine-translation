from tensorflow import keras
from keras import layers

"""### GRU"""

def create_gru_model(vocab_size, embed_dim, latent_dim, dropout, optimizer):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_gru = layers.GRU(latent_dim,
                             return_state=True,
                             name="encoder_gru")
    _, h = encoder_gru(x)
    encoder_states = [h]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_gru = layers.GRU(latent_dim,
                             return_sequences=True,
                             return_state=True,
                             name="decoder_gru")
    decoder_outputs, _ = decoder_gru(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, h = decoder_gru(x, initial_state=decoder_states_inputs)
    decoder_states = [h]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### GRU with pretrained embeddings"""

def create_gru_glove_model(vocab_size, embed_dim, latent_dim, dropout, optimizer, embeddings):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          embeddings_initializer=keras.initializers.Constant(embeddings),
                                          trainable=False,
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_gru = layers.GRU(latent_dim,
                             return_state=True,
                             name="encoder_gru")
    _, h = encoder_gru(x)
    encoder_states = [h]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_gru = layers.GRU(latent_dim,
                             return_sequences=True,
                             return_state=True,
                             name="decoder_gru")
    decoder_outputs, _ = decoder_gru(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, h = decoder_gru(x, initial_state=decoder_states_inputs)
    decoder_states = [h]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### Bidirectional GRU"""

def create_bi_gru_model(vocab_size, embed_dim, latent_dim, dropout, optimizer):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_gru = layers.Bidirectional(
        layers.GRU(latent_dim,
                   return_state=True, 
                   name="encoder_gru"),
        merge_mode="sum")
    _, forward_h, backward_h = encoder_gru(x)
    h = layers.Concatenate()([forward_h, backward_h])
    encoder_states = [h]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_gru = layers.GRU(2*latent_dim, 
                             return_sequences=True, 
                             return_state=True, 
                             name="decoder_gru")
    
    decoder_outputs, _ = decoder_gru(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(2*latent_dim,), name="decoder_state_input_h")
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, h = decoder_gru(x, initial_state=decoder_states_inputs)
    decoder_states = [h]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### Bidirectional GRU with pretrained embeddings"""

def create_bi_gru_glove_model(vocab_size, embed_dim, latent_dim, dropout, optimizer, embeddings):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          embeddings_initializer=keras.initializers.Constant(embeddings),
                                          trainable=False,
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_gru = layers.Bidirectional(
        layers.GRU(latent_dim, 
                   return_state=True, 
                   name="encoder_gru"),
        merge_mode="sum")
    _, forward_h, backward_h = encoder_gru(x)
    h = layers.Concatenate()([forward_h, backward_h])
    encoder_states = [h]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_gru = layers.GRU(2*latent_dim, 
                             return_sequences=True, 
                             return_state=True, 
                             name="decoder_gru")
    decoder_outputs, _ = decoder_gru(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(2*latent_dim,), name="decoder_state_input_h")
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, h = decoder_gru(x, initial_state=decoder_states_inputs)
    decoder_states = [h]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### LSTM"""

def create_lstm_model(vocab_size, embed_dim, latent_dim, dropout, optimizer):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_lstm = layers.LSTM(latent_dim,
                               return_state=True,
                               name="encoder_lstm")
    _, h, c = encoder_lstm(x)
    encoder_states = [h, c]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim,
                               return_sequences=True,
                               return_state=True,
                               name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, h, c = decoder_lstm(x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### LSTM with pretrained embeddings"""

def create_lstm_glove_model(vocab_size, embed_dim, latent_dim, dropout, optimizer, embeddings):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          embeddings_initializer=keras.initializers.Constant(embeddings),
                                          trainable=False,
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_lstm = layers.LSTM(latent_dim,
                               return_state=True,
                               name="encoder_lstm")
    _, h, c = encoder_lstm(x)
    encoder_states = [h, c]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim,
                               return_sequences=True,
                               return_state=True,
                               name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, h, c = decoder_lstm(x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### Bidirectional LSTM"""

def create_bi_lstm_model(vocab_size, embed_dim, latent_dim, dropout, optimizer):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_lstm = layers.Bidirectional(
        layers.LSTM(latent_dim,
                    return_state=True,
                    name="encoder_lstm"),
        merge_mode="sum")
    _, forward_h, forward_c, backward_h, backward_c = encoder_lstm(x)
    h = layers.Concatenate()([forward_h, backward_h])
    c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [h, c]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_lstm = layers.LSTM(2*latent_dim,
                               return_sequences=True, 
                               return_state=True,
                               name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(2*latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(2*latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, h, c = decoder_lstm(x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder

"""### Bidirectional LSTM with pretrained embeddings"""

def create_bi_lstm_glove_model(vocab_size, embed_dim, latent_dim, dropout, optimizer, embeddings):
    # endocer
    encoder_inputs = keras.Input(shape=(None,), 
                                 dtype="int64", 
                                 name="encoder_inputs")
    encoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          embeddings_initializer=keras.initializers.Constant(embeddings),
                                          trainable=False,
                                          mask_zero=True, 
                                          name="encoder_embeddings")
    x = encoder_embeddings(encoder_inputs)
    encoder_lstm = layers.Bidirectional(
        layers.LSTM(latent_dim,
                    return_state=True,
                    name="encoder_lstm"),
        merge_mode="sum")
    _, forward_h, forward_c, backward_h, backward_c = encoder_lstm(x)
    h = layers.Concatenate()([forward_h, backward_h])
    c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [h, c]
    encoder = keras.Model(encoder_inputs, encoder_states)

    # decoder for end-to-end model
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(vocab_size, 
                                          embed_dim, 
                                          mask_zero=True, 
                                          name="decoder_embeddings")
    x = decoder_embeddings(decoder_inputs)
    decoder_lstm = layers.LSTM(2*latent_dim,
                               return_sequences=True, 
                               return_state=True,
                               name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(dropout)(decoder_outputs)
    decoder_dense = layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # stand-alone decoder for making inference
    decoder_state_input_h = keras.Input(shape=(2*latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = keras.Input(shape=(2*latent_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, h, c = decoder_lstm(x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder, decoder