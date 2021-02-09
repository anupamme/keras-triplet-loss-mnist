import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def accuracy(predictions, values):
    correct = 0.0
    for prediction, value in zip(predictions, values):
        if prediction - value == 0.0:
            correct += 1
    
    return correct / len(predictions)

def convert_to_vocab(float_str):
    out = []
    for ch in float_str:
        if ch == '.':
            out.append(10)
        elif ch == '-':
            out.append(11)
        else:
            out.append(int(ch))
    if len(out) < 10:
        out.append(10)  # append .
        while len(out) < 10:
            out.append(0)   # keep appending 0
    return out

def get_loss_em(is_triplet=False):
    if is_triplet:
#        return tfa.losses.contrastive_loss()
        return tfa.losses.TripletSemiHardLoss()
    else:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def get_optimiser_em():
    return tf.keras.optimizers.Adam(0.001)

def get_model_em(batch_size, vocab_size=12, embedding_size=512, variables=1, bidirectional=True, share_embeddings=True):
    lstm_features = 512
    if share_embeddings:
        embedding_layer = tf.keras.layers.Embedding(
            vocab_size, embedding_size, embeddings_initializer='uniform',
            input_length=None
        )
        inputs = tf.keras.layers.Input(shape=(variables, None, ), name='input', dtype='int64')
        embeddings = embedding_layer(inputs)
        print(embeddings)
        embeddings = tf.unstack(embeddings, variables, axis=1)
        print(embeddings)
        
        lstm_outs = []
        for i in range(variables):
            if bidirectional:
                lstm_outs.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_features)) (embeddings[i]))
            else:
                lstm_outs.append(tf.keras.layers.LSTM(lstm_features) (embeddings[i]))
        print(lstm_outs)
        lstm_out = tf.stack(lstm_outs, 1)
        print('variables', variables)
        lstm_out = tf.reshape(lstm_out, [-1, variables * lstm_features])
        predictions = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(lstm_out)

        # if bidirectional:
        #     lstm_outs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_features)) (embeddings)
        # else:
        #     lstm_outs = tf.keras.layers.LSTM(lstm_features) (embeddings)
        # layer1 = tf.keras.layers.Dense(128, activation='relu')(lstm_outs)
        # predictions = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(lstm_outs)
    
    # # Build model
    return tf.keras.models.Model(inputs=inputs, outputs=predictions), lstm_out

def loss_reg(model, x, y, training, loss_object, model_em):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(np.asarray([x]), training=training)
    y_np = y_.numpy()
    _total_loss = 0
    for item in y_np[0]:
        val_1 = convert_to_vocab(str(item[0]))[:10]  #FIXME: make 10 a global variable xrange
        val_2 = convert_to_vocab(str(item[1]))[:10]
        val = [[val_1, val_2]]
        prob_vec = model_em.predict(val)
        print('actual values: ' + str(item))
        print('predicted probability: ' + str(prob_vec[0]))
        prob_non_sat = prob_vec[0][0]
        _val = loss_object(y_true=y, y_pred=y_).numpy()
        print('loss_local, regularising_loss: ' + str(_val) + ', ' + str(prob_non_sat))
        _total_loss = _total_loss + (float)(_val + prob_non_sat)
    return tf.convert_to_tensor(_total_loss)

def loss(model, x, y, training, loss_object, model_em):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(np.asarray([x]), training=training)
    _total_loss = loss_object(y_true=y, y_pred=y_)
    return _total_loss

def grad(model, inputs, targets, loss_object, model_em):
    with tf.GradientTape() as tape:
        loss_value = loss_reg(model, inputs, targets, True, loss_object, model_em)
    import pdb
    pdb.set_trace()
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def model_fit(model, x_train, y_train, loss_obj, optimizer_obj, model_em, epochs=1, batch_size=32):
    for e in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            # print(batch_x)
            batch_y = y_train[i : i + batch_size]
            loss_value, grads = grad(model, batch_x, batch_y, loss_obj, model_em)
            optimizer_obj.apply_gradients(zip(grads, model.trainable_variables))