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

# def loss_reg(model, x, y, training, loss_object, constraint_encoder):
#     # training=training is needed only if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     y_ = model(np.asarray([x]), training=training)
#     y_np = y_.numpy()
#     _total_loss = 0

#     # batch = []
#     # for item in y_np[0]:
#     #     query = [convert_to_vocab(str(item[0]))[:10], convert_to_vocab(str(item[1]))[:10]] #FIXME: make 10 a global variable xrange
#     #     batch.append(query)
    
#     # probs = constraint_encoder.predict(batch)

#     for item in y_np[0]:
#         val_1 = convert_to_vocab(str(item[0]))[:10]  #FIXME: make 10 a global variable xrange
#         val_2 = convert_to_vocab(str(item[1]))[:10]
#         val = [[val_1, val_2]]
#         prob_vec = constraint_encoder.predict(val)
#         print('actual values: ' + str(item))
#         print('predicted probability: ' + str(prob_vec[0]))
#         prob_non_sat = prob_vec[0][0]
#         _val = loss_object(y_true=y, y_pred=y_)
#         print('loss_local, regularising_loss: ' + str(_val) + ', ' + str(prob_non_sat))
#         # _total_loss = _total_loss + (float)(_val + prob_non_sat)
#         _val.assign_add(prob_non_sat)
#         return _val
#     return tf.convert_to_tensor(_total_loss)


def loss(model, x, y, training, loss_object):
    """[summary]

    Args:
        model ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]
        training ([type]): [description]
        loss_object ([type]): [description]

    Returns:
        [type]: [description]
    """
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def constrained_loss(model, x, y, training, loss_obj, constraint_loss):
    """[summary]

    Args:
        model ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]
        training ([type]): [description]
        loss_obj ([type]): [description]
        constraint_loss ([type]): [description]

    Returns:
        [type]: [description]
    """
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions, probs = model(x, training=training)
    loss = loss_obj(y_true=y[0], y_pred=predictions)
    
    # Calculate contraint penalties
    # print(probs)
    cl = constraint_loss(y_true=y[1], y_pred=probs)
    # print(tf.math.exp(cl))
    formula1 = tf.multiply(loss, tf.math.exp(cl)) # could use 2 instead of e as the base
    # print('formula1', formula1)
    formula2 = tf.add(loss, tf.multiply(loss, cl))
    # print('formula2', formula2)
    
    loss_constrained = formula1
    # Either sum or average the losses. Here we average
    loss_constrained = tf.math.reduce_mean(loss_constrained)
    return loss_constrained


def grad(model, inputs, targets, loss_object, constraint_loss=None):
    """[summary]

    Args:
        model ([type]): [description]
        inputs ([type]): [description]
        targets ([type]): [description]
        loss_object ([type]): [description]
        constraint_loss ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    with tf.GradientTape() as tape:
        if constraint_loss is not None:
            loss_value = constrained_loss(model, inputs, targets, True, loss_object, constraint_loss)
        else:
            loss_value = loss(model, inputs, targets, True, loss_object)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def model_fit(constrained_model, model, constraint_encoder, x_train, y_train, loss, optimizer, epochs=1, batch_size=32):
    """[summary]

    Args:
        constrained_model ([type]): [description]
        model ([type]): [description]
        constraint_encoder ([type]): [description]
        x_train ([type]): [description]
        y_train ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        epochs (int, optional): [description]. Defaults to 1.
        batch_size (int, optional): [description]. Defaults to 32.
    """
    for e in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            # print(batch_x)
            batch_y = y_train[i : i + batch_size]

            predictions = model.predict(batch_x)
            # Convert predictions to input format
            const_inputs = [[convert_to_vocab(str(prediction[0]))[:10], convert_to_vocab(str(prediction[1]))[:10]] for prediction in predictions]
            
            for l in range(len(constraint_encoder.layers)):
                constraint_encoder.layers[l].trainable = False
            
            # Debugging code - Do not delete for now
            # constraint_penalties = constraint_encoder.predict(const_inputs)
            # print(constraint_penalties)

            # prediction = constrained_model.predict([np.asarray(batch_x), np.asarray(const_inputs)])
            # print('prediction', prediction)
            
            # We do not train this. I just coded it for debugging - do not delete for now
            # mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
            # loss_value, grads = grad(model, np.asarray(batch_x), np.asarray(batch_y), mae)
            # print(loss_value)
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))

            batch_input = [np.asarray(batch_x), np.asarray(const_inputs)]
            batch_out = [np.asarray(batch_y), np.ones(len(batch_x))]
            constraint_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            loss_value, grads = grad(constrained_model, batch_input, batch_out, loss, constraint_loss)
            print('constrained_loss=', loss_value)
            # print(constrained_model.trainable_variables) # Double checking that the encoder is not being retrained.
            optimizer.apply_gradients(zip(grads, constrained_model.trainable_variables))
            # break

