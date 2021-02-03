'''
Different parameters:
1. Constraint_type: Monotonic, Synthetic
2. Training Data generation
3. Loss function: SVM, Triplet, Contrastive
4. Train a classifier and extract embeddings

Main Task
1. Pick a task: predict solubility of oxygen, transprecision computing, synthetic task
2. Train a baseline NN
3. Train a regularised NN
4. Compare.

Returns:
1. Embeddings
'''

from enum import Enum
import numpy as np
import math
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class Task_Type(Enum):
    Oxygen_Solubility = 1
    Transprecision_Computing = 2
    Synthetic_Task = 3
    
class Loss_Type(Enum):
    MSE = 1
    Triplet = 2
    Constrastive = 3
    
class Constraint_Type(Enum):
    Synthetic = 1
    Monotonic = 2
    Inequality = 3
    Multiple_Inequlity = 4

'''
synthetic formula: f(x1) = k1x1^2 + a1cos(p1pix1) + K
a1=0.3, p1=3, k1=1, K = 0.7
'''    
def compute_f(x):
    try:
        return 1*math.pow(x, 2) + 0.3*math.cos(3 * math.pi * x) + 0.7
    except TypeError as e:
        print(e)
    
'''

sample x1 from Gaussian

For monotonic constraint:
(x11, x12) X1 -> y1
(x21, x22) X2 -> y2

if X1 < X2, Y1 < Y2
Training data:

(X1, X2) such that X1 < X2
(Y1, Y2) such that Y1 < Y2

Sample x1 from Gaussian (mean, dev)
Sample/Choose x2 such that x2 > x1

Calculate Y1 and Y2

Enfore the constraint Y1 < Y2 (using Triplet loss/Contrastive Loss)
'''    
def create_synthetic_dataset_main(constraint_type):
    if constraint_type == Constraint_Type.Monotonic:
        x1 = np.random.normal(10, 0.1, 2000)
        x2 = list(map(lambda x: x + 1, x1))
        y1 = list(map(lambda x: compute_f(x), x1))
        y2 = list(map(lambda x: compute_f(x), x2))
        y3 = []
        x3 = []
        for idx, item in enumerate(x1):
            x3.append((item, x2[idx]))
        for idx, item in enumerate(y1):
            y3.append((item, y2[idx]))
        return (x3[:1500], y3[:1500]), (x3[1500:], y3[1500:])
    else:
        pass

def convert_to_vocab(float_str):
    out = []
    for ch in float_str:
        if ch == '.':
            out.append(10)
        else:
            out.append(int(ch))
    return out
    
'''
Range of yi: (100.21463122690456, 120.57669813450471)

output: 
    (y1, y2), 0
    
    y_values: 
'''    
def create_synthetic_dataset_em(y_train, maxlen=20000, input_dim=2, x_range=10):
    count = 0
    x_train = []
    y_train = []
    while count < maxlen:
        count += 1
        x_train_item = np.random.uniform((50, 150), input_dim)
        x_train_item_v = [convert_to_vocab(str(x_train_item[0]))[:x_range], convert_to_vocab(str(x_train_item[1]))[:x_range]]
        x_train.append(x_train_item_v)
        if (x_train_item[0] < x_train_item[1]):
            y_train_item = 1
        else:
            y_train_item = 0
        y_train.append(y_train_item)
    # convert to vocabulary
    
    return (x_train[:int(maxlen/2)], y_train[:int(maxlen/2)]), (x_train[int(maxlen/2):], y_train[int(maxlen/2):])
    
def get_model_em(batch_size, vocab_size=11, embedding_size=512, variables=1, bidirectional=True, share_embeddings=True):
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

def get_loss_em():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def get_optimiser_em():
    return tf.keras.optimizers.Adam(0.001)
'''
 q1. If task is transprecision computing:
1. use the data which comes as part of the codebase
elif task == oxygen solubility:
    use the data which comes with it.
elif task == synthetic constraint:
    create synthetic data
'''    
def create_synthetic_training_data(task_type, constraint_type):
    if task_type == Task_Type.Synthetic_Task:
        if constraint_type == Constraint_Type.Synthetic:
            pass
        elif constraint_type == Constraint_Type.Monotonic:
            pass
        elif constraint_type == Constraint_Type.Inequality:
            pass
        elif constraint_type == Constraint_Type.Multiple_Inequlity:
            pass
    pass

'''
1. create training data
2. define the network
3. define the loss function
4. call the training
5. return the model
'''
def create_embeddings(task_type, constraint_type, loss_fn_type):
    pass

'''
Input: embeddings_reference_point, classification_model, input_data
output: distance_from_thre_reference_point.

method: compute embeddings for the input_data and find its distnace from the reference point.
'''
def calculate_regularising_loss(y_pred, model_em):
    probability_val = model_em.predict(y_pred)
    return probability_val

'''
model to learn the formula. linear regressor
'''
#def get_model():
#    normalizer = preprocessing.Normalization()
#    linear_model = tf.keras.Sequential([
#        normalizer,
#        layers.Dense(units=1)
#    ])
#    return linear_model

# get the model
def get_model(n_inputs, n_outputs):
	model = tf.keras.Sequential()
	model.add(layers.Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(layers.Dense(n_outputs))
#	model.compile(loss='mae', optimizer='adam')
	return model

def get_loss_main():
    _obj = tf.keras.losses.MeanSquaredError
    return _obj

def get_optimiser():
    return tf.keras.optimizers.Adam(0.001)

def loss(model, x, y, training, loss_object, model_em):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    _total_loss = (float)(loss_object(y_true=y, y_pred=y_) / calculate_regularising_loss(y_, model_em))
    return _total_loss

def grad(model, inputs, targets, loss_object, model_em):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, True, loss_object, model_em)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def model_fit(model, x_train, y_train, loss_obj, optimizer_obj, model_em):
    for idx, x in enumerate(x_train):
        y = y_train[idx]
        # Optimize the model
        loss_value, grads = grad(model, x, y, loss_obj, model_em)
        optimizer_obj.apply_gradients(zip(grads, model.trainable_variables))
    pass

def accuracy(predictions, values):
    correct = 0.0
    for prediction, value in zip(predictions, values):
        if prediction - value == 0.0:
            correct += 1
    
    return correct / len(predictions)

def train_secondary_network(y_train):
    (x_train_em, y_train_em), (x_test_em, y_test_em) = create_synthetic_dataset_em(y_train)
    batch_size = 32
    maxlen = batch_size * 500
    variables = 2
    model_em, _ = get_model_em(batch_size, bidirectional=False, variables=variables)
    
    # with random inits
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    print(predictions)
    acc = accuracy(predictions, y_test_em)
    print('Accuracy Before:', acc)
    
    loss_em = get_loss_em()
    optimizer_em = get_optimiser_em()
    model_em.compile(
        optimizer=optimizer_em,
        loss=loss_em)
    
    model_em.fit(x_train_em, y_train_em, batch_size=32, epochs=1)
    
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    print(predictions)
    acc = accuracy(predictions, y_test_em)
    print('Accuracy After:', acc)
    return model_em

'''
NN to train solubility of oxygen: steps:
1. get training data 
2. Get NN architecture
3. write the training routine:
    write custom loss
4. 
5. 
'''
def train_main_network(x_train, y_train, x_val, y_val, model_em):
    # step 1: create embeddings
    # create training dataset for the constraint
    # Y1 < Y2 sample real valued uniformly distributed in -100 to 100
    
    
    # main network.
    model_1 = get_model(2, 2)
    model_2 = get_model(2, 2)
    model_1.compile(loss='mae', optimizer='adam')
    test_loss = model_1.evaluate(x_val, y_val)
    print('test loss 1 before: ' + str(test_loss))
    model_1.fit(x_train, y_train, batch_size=32, epochs=1)
    test_loss = model_1.evaluate(x_val, y_val)
    print('test loss 1 after: ' + str(test_loss))
    
    loss_obj = tf.keras.losses.MeanAbsoluteError()
    optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
    import pdb
    pdb.set_trace()
    model_fit(model_2, x_train, y_train, loss_obj, optimizer_obj, model_em)
    test_loss = model.evaluate(x_val, y_val)
    print('test loss: ' + str(test_loss))
    
if __name__=="__main__":
    (x_train, y_train), (x_val, y_val) = create_synthetic_dataset_main(Constraint_Type.Monotonic)
    model_em = train_secondary_network(y_train)
#    model_em = None
    train_main_network(x_train, y_train, x_val, y_val, model_em)