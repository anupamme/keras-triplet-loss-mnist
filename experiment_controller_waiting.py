import sys
import json
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
import numpy as np
import pandas as pd

from utils import util as u

def does_satisfay(y_val):
    return y_val < 445 and y_val > 0
        

def calculate_constraint_violation(model, x_test):
    y_pred = model.predict(x_test)
    num_violations = 0
    for item in y_pred:
        if not does_satisfay(item):
            num_violations += 1
    return num_violations

def create_synthetic_dataset_sec(y_train, maxlen=20000, input_dim=1, x_range=10):
    count = 0
    x_train = []
    y_train = []
    while count < maxlen:
        count += 1
        x_train_item = np.asarray([np.random.randint(0, 1000)])
        x_train_item_v = [u.convert_to_vocab(str(x_train_item[0]))[:x_range]]
        x_train.append(x_train_item_v)
        if does_satisfay(x_train_item[0]):
            y_train_item = 1
        else:
            y_train_item = 0
        y_train.append(y_train_item)
    # convert to vocabulary
    
    return (x_train[:int(maxlen/2)], y_train[:int(maxlen/2)]), (x_train[int(maxlen/2):], y_train[int(maxlen/2):])

def get_model(n_inputs, n_outputs, normalizer):
    model = tf.keras.Sequential()
#    model.add(normalizer)
    model.add(layers.Dense(64, input_dim=n_inputs, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(n_outputs))
    return model

def build_vocab(x1_data):
    vocab = {}
    count = 0
    for outer_id in x1_data:
        for inner_id in x1_data[outer_id]:
            if inner_id not in vocab:
                vocab[inner_id] = count
                count = count + 1
    return vocab

def train_secondary_network(y_train):
    (x_train_em, y_train_em), (x_test_em, y_test_em) = create_synthetic_dataset_sec(y_train)
    batch_size = 32
    maxlen = batch_size * 500
    variables = 1
    model_em, _ = u.get_model_em(batch_size, bidirectional=False, variables=variables)
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    print(predictions)
    acc = u.accuracy(predictions, y_test_em)
    print('Accuracy Before:', acc)
    
    loss_em = u.get_loss_em()
    optimizer_em = u.get_optimiser_em()
    model_em.compile(
        optimizer=optimizer_em,
        loss=loss_em)
    
    model_em.fit(x_train_em, y_train_em, batch_size=32, epochs=1)
    
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    print(predictions)
    acc = u.accuracy(predictions, y_test_em)
    print('Accuracy After:', acc)
    return model_em
    

def create_dataset_main(x_1_file, x_2_file, y_file):
    x1_data = json.loads(open(x_1_file, 'r').read())
    x2_data = json.loads(open(x_2_file, 'r').read())
    y_data = json.loads(open(y_file, 'r').read())
    vocab_space_map = build_vocab(x1_data)
    training_data_x = []
    training_data_y = []
    for time_idx in x1_data:
        for space_idx in x1_data[time_idx]:
            space_id_vocab = vocab_space_map[space_idx]
            x1_val = x1_data[time_idx][space_idx]
            if (time_idx not in x2_data) or (space_idx not in x2_data[time_idx]):
                x2_val = 0 
            else:
                x2_val = x2_data[time_idx][space_idx]
            training_data_x.append((int(time_idx), space_id_vocab, x1_val, x2_val))
            if (time_idx not in y_data) or (space_idx not in y_data[time_idx]):
                y_val = 0
            else:
                y_val = y_data[time_idx][space_idx]
            training_data_y.append(y_val)
    train_x = training_data_x[:int(len(training_data_x)/2)]
    train_y = training_data_y[:int(len(training_data_y)/2)]
    val_x = training_data_x[int(len(training_data_x)/2):int(3*len(training_data_x)/4)]
    val_y = training_data_y[int(len(training_data_y)/2):int(3*len(training_data_y)/4)]
    test_x = training_data_x[int(3*len(training_data_x)/4):]
    test_y = training_data_y[int(3*len(training_data_y)/4):]
    return (train_x, train_y),(val_x, val_y),(test_x, test_y)

def train_main_network(x_train, y_train, x_val, y_val, x_test, y_test, constraint_encoder):
    
#    normalizer = preprocessing.Normalization()
#    normalizer.adapt(x_train)
    
#    model = get_model(4, 1, normalizer)
    ffn_model = u.create_FFN([4, 64, 64, 64, 1], ['tanh', 'tanh', 'tanh', 'tanh'])
    ffn_model_copy = tf.keras.models.clone_model(ffn_model)
    print(ffn_model.outputs[0].shape)
    
#    test_loss = ffn_model.evaluate(x_val, y_val)
#    print('test loss before: ' + str(test_loss))
    
    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    
    ffn_model.compile(optimizer=optimizer_obj, loss=mae)
    ffn_model_copy.compile(optimizer=optimizer_obj, loss=mae)

    test_loss = ffn_model_copy.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(ffn_model_copy, x_test)
    print('MAE, num violations before constraint training: ' + str(test_loss) + ', ' + str(num_violations))
    
    history = ffn_model_copy.fit(x_train, y_train, batch_size=32, epochs=1)
    test_loss = ffn_model_copy.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(ffn_model_copy, x_test)
    print('MAE, num violations without constraint training: ' + str(test_loss) + ', ' + str(num_violations))
    
    constrained_model = u.create_constrained_model(ffn_model, constraint_encoder)
    constrained_model, best_model = u.model_fit(constrained_model, ffn_model, constraint_encoder, x_train, y_train, x_val, y_val, mae, optimizer_obj, epochs=1)
    
    best_model.compile(optimizer=optimizer_obj, loss=mae)
    test_loss = best_model.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(best_model, x_test)
    print('Best MAE, num violations with constraint training: ' + str(test_loss) + ', ' + str(num_violations))
    
    test_loss = ffn_model.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(ffn_model, x_test)
    print('MAE, num violations with constraint training: ' + str(test_loss) + ', ' + str(num_violations))
    
    x_ds = pd.DataFrame(x_train)
    x_test_ds = pd.DataFrame(x_test)
    y_ds = pd.DataFrame(y_train)
    print(x_ds.describe())
    print(x_test_ds.describe())
    print(y_ds.describe())
    
#    print(len(history.history['loss']))
    #    model_fit(model, x_train, y_train, loss_obj, optimizer_obj)
#    test_loss = model.evaluate(x_val, y_val)
#    print('test loss after: ' + str(test_loss))
#    print(model.summary())
    
    pass

if __name__=="__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = create_dataset_main(sys.argv[1], sys.argv[2], sys.argv[3])
    model_em = train_secondary_network(y_train)
#    model_em = None
    train_main_network(x_train, y_train, x_val, y_val, x_test, y_test, model_em)