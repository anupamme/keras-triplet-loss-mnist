import io
import numpy as np
import pdb
import sys

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

'''
Q: what is variables?
'''
def get_char_LSTM(batch_size, vocab_size=11, embedding_size=512, variables=1, bidirectional=True, share_embeddings=True):
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


def loss(model, x, y, training, loss_object):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    # print('x:', x, x.shape)
    # print('y:', y_)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, True, loss_object)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def model_fit(model, x_train, y_train, loss_object, optimizer, epochs=1):
    # pdb.set_trace()
    # Create batches
    for e in range(epochs):
        for i in range(0, len(x_train), 32):
            batch_x = x_train[i : i + 32]
            # print(batch_x)
            batch_y = y_train[i : i + 32]
            loss_value, grads = grad(model, batch_x, batch_y, loss_object)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # for idx, x in enumerate(x_train):
    #     print(x)
    #     y = y_train[idx]
    #     # Optimize the model
    #     loss_value, grads = grad(model, x, y, loss_object)
    #     print(loss_value)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))

def accuracy(predictions, values):
    correct = 0.0
    for prediction, value in zip(predictions, values):
        if prediction - value == 0.0:
            correct += 1
    
    return correct / len(predictions)


def main():
    batch_size = 32
    maxlen = batch_size * 500
    variables = 2
    x_dim = 2
    _type = sys.argv[1]
    # input_size = 2
    if _type == 'func':
#        model = get_model_LSTM(10)
        num_var = 1
        model, lstm_out = get_char_LSTM(batch_size, bidirectional=False, variables=variables)
        print('lstm_out:', lstm_out)
        x_train = np.random.randint(11, size=(maxlen, variables, x_dim))
        y_train = [1 if np.sum(x) < 40 else 0 for x in x_train]
        print(x_train[: batch_size], [np.sum(x) for x in x_train[: batch_size]])
        print(y_train[: 50])
        # y_train = np.random.randint(2, size=(maxlen))
        # print(x_train)
        p = model.predict(x_train)
        print(p[: batch_size])

        x_test = np.random.randint(11, size=(batch_size * 100, variables, x_dim))
        y_test = [1 if np.sum(x) < 40 else 0 for x in x_test]
        # y_test = np.random.randint(2, size=(32))
        import pdb
        pdb.set_trace()
        print('y_test', y_test)
        prediction_probs = model.predict(x_test)
        predictions = [int(np.round(p[1])) for p in prediction_probs]
        print(prediction_probs)
        print(predictions)
        acc = accuracy(predictions, y_test)
        print('Accuracy:', acc)
    
#   loss_obj = tfa.losses.TripletSemiHardLoss()
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer_obj = tf.keras.optimizers.Adam(0.001)
    # Compile the model
#    print(model.summary())
#    model.compile(
#        optimizer=tf.keras.optimizers.Adam(0.001),
#        loss=tfa.losses.TripletSemiHardLoss())
#    model.compile(
#        optimizer=tf.keras.optimizers.Adam(0.001),
#        loss=tf.keras.losses.BinaryCrossentropy())
    if _type == 'func' or _type == 'func_star':
        for e in range(3):
            model_fit(model, x_train, y_train, loss_obj, optimizer_obj)
    #        history = model.fit(x_train, y_train, batch_size=32, epochs=8)
            prediction_probs = model.predict(x_test)
            predictions = [int(np.round(p[1])) for p in prediction_probs]
            print(prediction_probs)
            print(predictions)
            acc = accuracy(predictions, y_test)
            print('Accuracy:', acc)
        
        import pdb
        pdb.set_trace()
        prediction_probs = model.predict(x_test)
        np.savetxt("vecs.tsv", prediction_probs, delimiter='\t')
        
    # Save test embeddings for visualization in projector
    # test_loss = model.evaluate(x_test, y_test)
    # print('test loss: ' + str(test_loss))
    # np.savetxt("vecs.tsv", results, delimiter='\t')

#    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#    for img, labels in tfds.as_numpy(test_dataset):
#        [out_m.write(str(x) + "\n") for x in labels]
#    out_m.close()


    # try:
    #     from google.colab import files
    #     files.download('vecs.tsv')
    #     files.download('meta.tsv')
    # except:
    #     pass


if __name__ == "__main__":
    main()
