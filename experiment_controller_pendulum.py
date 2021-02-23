import numpy as np
import math
import tensorflow as tf

from utils import util as u

gravity = 9.81
length_pendulum = 1
theta_not = np.random.uniform(5, 6)
    
def does_satisfay(y_1, y_2):
    min_dif = 0.01
    return abs(math.pow(y_1, 2) + math.pow(y_2, 2) - math.pow(length_pendulum, 2)) < min_dif

def calculate_constraint_violation(model, x_test):
    y_pred = model.predict(x_test)
    num_violations = 0
    for item in y_pred:
        if not does_satisfay(item[0], item[1]):
            num_violations += 1
    return num_violations    
'''
theta << 1

theta(t) = theta_not * cos((g/l)^0.5 * t)

theta_not = 6 degrees
g = gravity = 9.8 m/s^2
l = length of pendulum = 1 m
t = 0 to infinity

x_train: theta
y_train: True/False
'''
def create_synthetic_dataset_em(maxlen=20000, x_range=10):
    count = 0
    x_train = []
    y_train = []
    while count < maxlen:
        # sample x, y, z
        theta_t = theta_not * math.cos(math.pow((gravity/length_pendulum), 0.5) * count)
        x_val = length_pendulum * math.sin(theta_t)
        y_val = length_pendulum * math.cos(theta_t)
        # add the label: with probability 0.5 add noise to x, y
        dice = np.random.uniform(0,1)
        if dice >= 0.5:
            noise = np.random.normal(0,1)
            x_val = x_val + noise
            y_val = y_val + noise
            y_label = 0
        else:
            y_label = 1
        count += 1
        x_train_item_v = [u.convert_to_vocab(str(x_val))[:x_range], u.convert_to_vocab(str(y_val))[:x_range]]
        x_train.append(x_train_item_v)
        y_train.append(y_label)
    return (x_train[:int(maxlen/2)], y_train[:int(maxlen/2)]), (x_train[int(maxlen/2):int(3*maxlen/4)], y_train[int(maxlen/2):int(3*maxlen/4)]), (x_train[int(3*maxlen/4):], y_train[int(3*maxlen/4):])

def train_secondary_network():
    (x_train_em, y_train_em), (x_val, y_val), (x_test_em, y_test_em) = create_synthetic_dataset_em()
    batch_size = 32
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
    
    model_em.fit(x_train_em, y_train_em, batch_size=batch_size, epochs=2, validation_data=(x_val, y_val))
    
    prediction_probs = model_em.predict(x_test_em)
    predictions = [int(np.round(p[1])) for p in prediction_probs]
    print(prediction_probs)
    # print(predictions)
    acc = u.accuracy(predictions, y_test_em)
    print('Accuracy After:', acc)
    return model_em

'''

sample x1 from Gaussian

x: x(t), y(t) | (t): z
'''    
def create_synthetic_dataset_main(data_size=10000):
    count = 0
    t = 0.1
    x_data = []
    y_data = []
    x_val = length_pendulum * math.sin(theta_not)
    y_val = length_pendulum * math.cos(theta_not)
    x_data.append((x_val, x_val))
    while count < data_size:
        # sample x, y, z
        theta_t = theta_not * math.cos(math.pow((gravity/length_pendulum), 0.5) * t)
        x_val = length_pendulum * math.sin(theta_t)
        y_val = length_pendulum * math.cos(theta_t)
        y_data.append((x_val, y_val))
        x_data.append((x_val, y_val))
        t = t + 0.1
        count = count + 1
    
    
    return (x_data[:int(data_size/2)], y_data[:int(data_size/2)]), (x_data[int(data_size/2):int(3*data_size/4)], y_data[int(data_size/2):int(3*data_size/4)]), (x_data[int(3*data_size/4):data_size], y_data[int(3*data_size/4):data_size])

def train_main_network(constraint_encoder):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = create_synthetic_dataset_main()
    # step 1: create embeddings
    # create training dataset for the constraint
    # Y1 < Y2 sample real valued uniformly distributed in -100 to 100
    ffn_model = u.create_FFN([2, 20, 2])
    ffn_model_copy = tf.keras.models.clone_model(ffn_model)
    print(ffn_model.outputs[0].shape)
    
    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    ffn_model.compile(optimizer=optimizer_obj, loss=mae)
    ffn_model_copy.compile(optimizer=optimizer_obj, loss=mae)
    
    test_loss = ffn_model_copy.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(ffn_model_copy, x_test)
    print('MAE, num violations before constraint training: ' + str(test_loss) + ', ' + str(num_violations) + ', ' + str(num_violations/len(x_test)))
    
    history = ffn_model_copy.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
    test_loss = ffn_model_copy.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(ffn_model_copy, x_test)
    print('MAE, num violations without constraint training: ' + str(test_loss) + ', ' + str(num_violations) + ', ' + str(num_violations/len(x_test)))
    
    constrained_model = u.create_constrained_model(ffn_model, constraint_encoder)
    constrained_model, best_model = u.model_fit(constrained_model, ffn_model, constraint_encoder, x_train, y_train, x_val, y_val, mae, optimizer_obj, epochs=2)
    
    best_model.compile(optimizer=optimizer_obj, loss=mae)
    test_loss = best_model.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(best_model, x_test)
    print('Best MAE, num violations with constraint training: ' + str(test_loss) + ', ' + str(num_violations) + ', ' + str(num_violations/len(x_test)))
    
    test_loss = ffn_model.evaluate(x_test, y_test)
    num_violations = calculate_constraint_violation(ffn_model, x_test)
    print('MAE, num violations with constraint training: ' + str(test_loss) + ', ' + str(num_violations) + ', ' + str(num_violations/len(x_test)))

if __name__=="__main__":
    constraint_encoder = train_secondary_network()
    train_main_network(constraint_encoder)