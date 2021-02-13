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

from utils import util as u

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
        x_train_item_v = [u.convert_to_vocab(str(x_train_item[0]))[:x_range], u.convert_to_vocab(str(x_train_item[1]))[:x_range]]
        x_train.append(x_train_item_v)
        if (x_train_item[0] < x_train_item[1]):
            y_train_item = 1
        else:
            y_train_item = 0
        y_train.append(y_train_item)
    # convert to vocabulary
    
    return (x_train[:int(maxlen/2)], y_train[:int(maxlen/2)]), (x_train[int(maxlen/2):], y_train[int(maxlen/2):])    

# get the model
def get_model(n_inputs, n_outputs):
	model = tf.keras.Sequential()
	model.add(layers.Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(layers.Dense(n_outputs))
#	model.compile(loss='mae', optimizer='adam')
	return model

def train_secondary_network(y_train):
    (x_train_em, y_train_em), (x_test_em, y_test_em) = create_synthetic_dataset_em(y_train)
    batch_size = 32
    maxlen = batch_size * 500
    variables = 2
    model_em, _ = u.get_model_em(batch_size, bidirectional=False, variables=variables)
    # with random inits
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    # print(predictions)
    acc = u.accuracy(predictions, y_test_em)
    print('Accuracy Before:', acc)
    
    loss_em = u.get_loss_em(is_triplet=False)
    optimizer_em = u.get_optimiser_em()
    model_em.compile(
        optimizer=optimizer_em,
        loss=loss_em)
    
    model_em.fit(x_train_em, y_train_em, batch_size=32, epochs=1)
    
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    # print(predictions)
    acc = u.accuracy(predictions, y_test_em)
    print('Accuracy After:', acc)
    return model_em

def accuracy(predictions, values):
    mae = 0.0
    for prediction, value in zip(predictions, values):
        mae += abs(prediction[0] - value[0]) + abs(prediction[1] - value[1])
    return mae

'''
NN to train solubility of oxygen: steps:
1. get training data 
2. Get NN architecture
3. write the training routine:
    write custom loss
4. 
5. 
'''
def train_main_network(x_train, y_train, x_val, y_val, constraint_encoder):
    # step 1: create embeddings
    # create training dataset for the constraint
    # Y1 < Y2 sample real valued uniformly distributed in -100 to 100
    ffn_model = u.create_FFN([2, 20, 2])
    print(ffn_model.outputs[0].shape)
    # print(x_train)
    y_predict = ffn_model.predict(x_val)
    acc = accuracy(y_predict, y_val)
    print('MAE before training: ' + str(acc))
    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
    ffn_model.compile(optimizer=optimizer_obj, loss=mae)
#    history = ffn_model.fit(x_train, y_train, batch_size=32, epochs=1)
#    test_loss = ffn_model.evaluate(x_val, y_val)
#    print('MAE without constraint training: ' + str(test_loss))
    constrained_model = u.create_constrained_model(ffn_model, constraint_encoder)
    constrained_model = u.model_fit(constrained_model, ffn_model, constraint_encoder, x_train, y_train, mae, optimizer_obj)
    test_loss = ffn_model.evaluate(x_val, y_val)
    print('MAE with constraint training: ' + str(test_loss))
    
    # losses = {
    #     "predictions" : tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)#,
    #     # "predictions" : tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # }
    # lossWeights = {"predictions": 1.0}

    # ffn_model.compile(
    #     optimizer=optimizer_obj,
    #     loss = losses, loss_weights=lossWeights
    #     # loss=[tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE),\
    #     #     tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)]
    # )
    
    # batch_size = 32
    # for e in range(1):
    #     for i in range(0, len(x_train), batch_size):
    #         batch_x = x_train[i : i + batch_size]
    #         batch_y = y_train[i : i + batch_size]
    #         print(batch_x)
    #         loss = ffn_model.train_on_batch(np.array(batch_x), np.array(batch_y))
    #         print('Loss:', loss)
    #         break

    # # losses = {
    # #     "predictions" : tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE),
    # #     "predictions" : tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    # # }
    # # lossWeights = {"predictions": 1.0}

    # # batch_size = 32
    # # for e in range(1):
    # #     for i in range(0, len(x_train), batch_size):
    # #         batch_x = x_train[i : i + batch_size]
    # #         batch_y = y_train[i : i + batch_size]
    # #         print(batch_x)

    # #         predictions = model.predict(batch_x)
    # #         # Convert predictions to input format
    # #         const_inputs = [[convert_to_vocab(str(prediction[0]))[:10], \
    # #             convert_to_vocab(str(prediction[1]))[:10]] for prediction in predictions]
            
    # #         loss = constrained_model.train_on_batch([np.asarray(batch_x), np.asarray(const_inputs)])
    # #         print('Loss:', loss)
    # #         break



if __name__=="__main__":
    (x_train, y_train), (x_val, y_val) = create_synthetic_dataset_main(Constraint_Type.Monotonic)
    constraint_encoder = train_secondary_network(y_train)
#    model_em = None
    train_main_network(x_train, y_train, x_val, y_val, constraint_encoder)