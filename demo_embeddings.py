import io
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras import layers

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

def get_data(data_type = 'mnist', data_kind='img'):
    train_dataset, test_dataset = tfds.load(name=data_type, split=['train', 'test'], as_supervised=True)
    # Build your input pipelines
    train_dataset = train_dataset.shuffle(1024).batch(32)
    test_dataset = test_dataset.batch(32)
    if data_kind == 'img':
        train_dataset = train_dataset.map(_normalize_img)
        test_dataset = test_dataset.map(_normalize_img)
    return train_dataset, test_dataset

def get_model(shape=(28,28,1)):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=shape),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings,
    ])

def get_model_LSTM():
    max_features = 20000  # Only consider the top 20k words
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Add a classifier
#    outputs = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(256, activation=None)(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def get_data_text():
    max_features = 200000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(
        num_words=max_features
    )
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
    
    return (x_train, y_train), (x_val, y_val)

'''
format: 
(x_train, y_train), (x_val, y_val)
x_train.shape: (10,): double
y_train.shape: (10,): bool

'''
def get_data_numerical():
    maxlen = 20000
    x_train = np.random.rand(maxlen, 1)
    y_train = np.zeros(x_train.shape)
    for id_r, item_r in enumerate(x_train):
        for id_c, item_c in enumerate(item_r):
            if item_c > 0.5:
                y_train[id_r][id_c] = 1
            else:
                y_train[id_r][id_c] = 0
    
    x_val = np.random.rand(int(maxlen/4), 1)
    y_val = np.zeros(x_val.shape)
    for id_r, item_r in enumerate(x_val):
        for id_c, item_c in enumerate(item_r):
            if item_c > 0.5:
                y_val[id_r][id_c] = 1
            else:
                y_val[id_r][id_c] = 0
                
    return (x_train, y_train), (x_val, y_val)
    
    
def main():
#    model = get_model()
#
#    model = get_model(shape=(500,500,3))
    
    model = get_model_LSTM()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())
#
#    model2.compile(
#        optimizer=tf.keras.optimizers.Adam(0.001),
#        loss=tfa.losses.TripletSemiHardLoss())

#    train_dataset, test_dataset = get_data()
#    train_dataset, test_dataset = get_data(data_type='beans')

#    (x_train, y_train), (x_val, y_val) = get_data_text()
    
    (x_train, y_train), (x_val, y_val) = get_data_numerical()
    test_dataset = (x_val, y_val)
    
#    model.compile(tf.keras.optimizers.Adam(0.001), tfa.losses.TripletSemiHardLoss(), metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=32, epochs=1)

    
    
    # Train the network
#    history = model.fit(
#        train_dataset,
#        epochs=1)
#
#    history2 = model2.fit(
#        train_dataset2,
#        epochs=1)


    # Evaluate the network
#    import pdb
#    pdb.set_trace()
#    print_dataset(test_dataset)
    
    results = model.predict(test_dataset)
    # Save test embeddings for visualization in projector
    np.savetxt("vecs.tsv", results, delimiter='\t')

#    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#    for img, labels in tfds.as_numpy(test_dataset):
#        [out_m.write(str(x) + "\n") for x in labels]
#    out_m.close()


    try:
        from google.colab import files
        files.download('vecs.tsv')
        files.download('meta.tsv')
    except:
        pass
    
def print_dataset(ds):
    for example in ds.take(1):
        image, label = example[0], example[1]
        

if __name__ == "__main__":
    main()