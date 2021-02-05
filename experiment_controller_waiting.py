import sys
import json
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from utils import util as u

def create_synthetic_dataset_sec(y_train, maxlen=20000, input_dim=1, x_range=10):
    count = 0
    x_train = []
    y_train = []
    while count < maxlen:
        count += 1
        x_train_item = [np.random.randint((0, 200), input_dim)]
#        x_train_item_v = [u.convert_to_vocab(str(x_train_item[0]))[:x_range]]
        x_train.append(x_train_item)
        if (x_train_item[0] < 150) and (x_train_item[0] > 50):
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
    test_x = training_data_x[int(len(training_data_x)/2):]
    test_y = training_data_y[int(len(training_data_y)/2):]
    return (train_x, train_y),(test_x, test_y)

def train_main_network(x_train, y_train, x_val, y_val, model_em):
    
    normalizer = preprocessing.Normalization()
    normalizer.adapt(x_train)
    model = get_model(4, 1, normalizer)
    test_loss = model.evaluate(x_val, y_val)
    print('test loss before: ' + str(test_loss))
    
    optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_obj = tf.keras.losses.MeanAbsoluteError()
    
#    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
#    model.compile(loss=loss_obj, optimizer=optimizer_obj)
#    history = model.fit(x_train, y_train, batch_size=32, epochs=2000, callbacks=[callback])
    
    u.model_fit(model, x_train, y_train, loss_obj, optimizer_obj, model_em)
    
    print(len(history.history['loss']))
    #    model_fit(model, x_train, y_train, loss_obj, optimizer_obj)
    test_loss = model.evaluate(x_val, y_val)
    print('test loss after: ' + str(test_loss))
    print(model.summary())
    
    pass

if __name__=="__main__":
    (x_train, y_train), (x_val, y_val) = create_dataset_main(sys.argv[1], sys.argv[2], sys.argv[3])
    model_em = train_secondary_network(y_train)
#    model_em = None
    train_main_network(x_train, y_train, x_val, y_val, model_em)