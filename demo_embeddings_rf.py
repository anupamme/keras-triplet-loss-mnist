import io
import numpy as np
import sys

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


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
   
    
def create_vocabulary_map(x_train, independent=False):
    assert len(x_train) > 0
    if len(x_train[0]) > 1:
        if independent:
            pass
        else:
            pass
    else:
        vocabulary_set = set()
        for number in x_train:
            vocabulary_set.add(number[0])
        num2id = {entry:i for i, entry in enumerate(vocabulary_set)}
        id2num = {i:entry for i, entry in enumerate(vocabulary_set)}
        return num2id, id2num


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
    return tf.keras.Sequential(
    [
        tf.keras.Input(shape=(None,), dtype="int32"),
        tf.keras.layers.Embedding(max_features, 128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(256, activation=None)
    ])

def get_model_FF():
    max_features = 20000  # Only consider the top 20k words
    return tf.keras.Sequential([
#        tf.keras.Input(shape=(None,), dtype="int32"),
        tf.keras.layers.Embedding(max_features, 128),
#        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(256, activation=None)
#        tf.keras.layers.Dense(256, input_dim=2, activation=None)
    ])
    

def get_data_text():
    max_features = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(
        num_words=max_features
    )
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
    
    return (x_train, y_train), (x_val, y_val)


def decide(val):
    if pow(val, 2) < 0.25 or pow(val, 2) > 0.75:
        return 1
    else:
        return 0
    
def decide_two(val1, val2):
    if (pow(val1, 2) + pow(val2, 2)) < 0.25 or (pow(val1, 2) + pow(val2, 2)) > 0.75:
        return 1
    else:
        return 0

'''
format: 
(x_train, y_train), (x_val, y_val)
x_train.shape: (10,): double
y_train.shape: (10,): bool


'''
def get_data_numerical():
    maxlen = 20000
    x_train = np.random.rand(maxlen, 1)
    for item in x_train:
        item[0] = round(item[0], 3)
    y_train = np.zeros(x_train.shape)
    for id_r, item_r in enumerate(x_train):
        y_train[id_r][0] = decide(item_r[0])
    
    x_val = np.random.rand(maxlen, 1)
    for item in x_val:
        item[0] = round(item[0], 3)
    y_val = np.zeros(x_val.shape)
    for id_r, item_r in enumerate(x_val):
        y_val[id_r][0] = decide(item_r[0])
                
    return (x_train, y_train), (x_val, y_val)
    
def get_data_numerical_two():
    maxlen = 20000
    x_train = np.random.rand(maxlen, 2)
    for item in x_train:
        item[0] = round(item[0], 4)
        item[1] = round(item[1], 4)
        
    y_train = np.zeros((maxlen, 1))
    for id_r, item_r in enumerate(x_train):
        y_train[id_r][0] = decide_two(item_r[0], item_r[1])
    
    x_val = np.random.rand(maxlen, 2)
    for item in x_val:
        item[0] = round(item[0], 3)
        item[1] = round(item[1], 3)
    y_val = np.zeros((maxlen, 1))
    for id_r, item_r in enumerate(x_val):
        y_val[id_r][0] = decide_two(item_r[0], item_r[1])
                
    return (x_train, y_train), (x_val, y_val)

def decide_var(val):
    sq_sum = 0
    for item in val:
        sq_sum += pow(item, 2)
    
    if sq_sum < 0.25 or sq_sum > 0.75:
        return 1
    else:
        return 0 

def get_data_numerical_meta(input_dim, decider_fn, maxlen=20000, rounding_limit=4):
    x_train = np.random.rand(maxlen, input_dim)
    x_val = np.random.rand(maxlen, input_dim)
    for item in x_train:
        dim_idx = 0
        while dim_idx < input_dim:
            item[dim_idx] = round(item[dim_idx], rounding_limit)
            dim_idx += 1
    for item in x_val:
        dim_idx = 0
        while dim_idx < input_dim:
            item[dim_idx] = round(item[dim_idx], rounding_limit)
            dim_idx += 1
    
    y_train = np.zeros((maxlen, 1))
    y_val = np.zeros((maxlen, 1))
    for id_r, item_r in enumerate(x_train):
        y_train[id_r][0] = decider_fn(item_r)
    for id_r, item_r in enumerate(x_val):
        y_val[id_r][0] = decider_fn(item_r)
    
    return (x_train, y_train), (x_val, y_val)


def convert_to_vocab(data_arr, mul_factor=10000):
    dest = np.empty_like(data_arr)
    for id_r, item_r in enumerate(data_arr):
        for id_c, item_c in enumerate(item_r):
            dest[id_r][id_c] = item_c * mul_factor
    return dest
    
def main():
    
    _type = sys.argv[1]
    
    if _type == 'mnist':
        model = get_model()
        train_dataset, test_dataset = get_data()
    elif _type == 'imdb':
        model = get_model_LSTM()
        (x_train, y_train), (x_val, y_val) = get_data_text()
        test_dataset = (x_val, y_val)
    elif _type == 'func':
        model = get_model_FF()
        (x_train, y_train), (x_val, y_val) = get_data_numerical_meta(2, decide_var)
        x_train = convert_to_vocab(x_train)
        x_val = convert_to_vocab(x_val)
        
        test_dataset = (x_val, y_val)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())
    if _type == 'mnist':
        history = model.fit(train_dataset, epochs=1)
        results = model.predict(test_dataset)
    elif _type == 'imdb':
        history = model.fit(x_train, y_train, batch_size=32, epochs=1)
        results = model.predict(test_dataset)
    elif _type == 'func':
        history = model.fit(x_train, y_train, batch_size=32, epochs=2)
        results = model.predict(x_val)
    
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
