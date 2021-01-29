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
Range of yi:

output: 
    (y1, y2), 0
    ...
    y_values: 
'''    
def create_synthetic_dataset_em(y_train, maxlen=10000, input_dim=2):
    x_train = np.random.rand(maxlen, input_dim)
    x_val = np.random.rand(maxlen, input_dim)
    
    pass
    
    
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
def calculate_regularising_loss(y_true, y_pred):
    pass

'''
model to learn the formula. linear regressor
'''
def get_model():
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    return linear_model

def get_loss_main():
    _obj = tf.keras.losses.MeanSquaredError
    return _obj

def get_optimiser():
    _obj = tf.keras.optimizers.Adam(0.001)

def loss(model, x, y, training, loss_object):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    _total_loss = loss_object(y_true=y, y_pred=y_) + calculate_regularising_loss(y_true=y, y_pred=y_)
    return _total_loss

def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, True, loss_object)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def model_fit(model, x_train, y_train, loss_obj, optimizer_obj):
    for idx, x in enumerate(x_train):
        y = y_train[idx]
        # Optimize the model
        loss_value, grads = grad(model, x, y, loss_object)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    pass

'''
NN to train solubility of oxygen: steps:
1. get training data 
2. Get NN architecture
3. write the training routine:
    write custom loss
4. 
5. 
'''
def train_main_network(task_type):
    (x_train, y_train), (x_val, y_val) = create_synthetic_dataset_main(Constraint_Type.Monotonic)
    # step 1: create embeddings
    # create training dataset for the constraint
    # Y1 < Y2 sample real valued uniformly distributed in -100 to 100
    (x_train_em, y_train_em), (x_val_em, y_val_em) = create_synthetic_dataset_em(y_train)
    model_em = get_model_em()
    loss_em = get_loss_em()
    optimizer_em = get_optimiser_em()
    model_em.fit(x_train, y_train, batch_size=32, epochs=1)
    
    model = get_model()
    loss_obj = get_loss_main()
    optimizer_obj = get_optimiser()
    model_fit(model, x_train, y_train, loss_obj, optimizer_obj)
    test_loss = model.evaluate(x_val, y_val)
    print('test loss: ' + str(test_loss))