import numpy as np
import math
import tensorflow as tf

from utils import util as u
'''

Input: X, Y, Z

For helical motion:
x^2 + y^2 = 1
z > 0

for x and y: sample values from a circular motion
for z: 

Add some noise to the value of x and y and add negative values of z: for negative sampling

x_train :: (x, y, z):
    sample the theta angle it makes with the axis: 0 < theta < 360
    
y_train :: (1, 0)
'''    
def create_synthetic_dataset_em(maxlen=20000, x_range=10):
    count = 0
    x_train = []
    y_train = []
    while count < maxlen:
        # sample x, y, z
        angle = np.random.uniform(0, 360)
        x_item = math.sin(angle) 
        y_item = math.cos(angle)
#        z_item = np.random.uniform(-100, 100)
        z_item = angle
        # add the label: with probability 0.5 add noise to x, y
        dice = np.random.uniform(0,1)
        if dice >= 0.5:
            noise = np.random.normal(0,1)
            x_item = x_item + noise
            y_item = y_item + noise
            y_label = 0
        else:
            y_label = 1
        count += 1
        x_train_item_v = [u.convert_to_vocab(str(x_item))[:x_range], u.convert_to_vocab(str(y_item))[:x_range], u.convert_to_vocab(str(z_item))[:x_range]]
        x_train.append(x_train_item_v)
        y_train.append(y_label)
    return (x_train[:int(maxlen/2)], y_train[:int(maxlen/2)]), (x_train[int(maxlen/2):int(3*maxlen/4)], y_train[int(maxlen/2):int(3*maxlen/4)]), (x_train[int(3*maxlen/4):], y_train[int(3*maxlen/4):])

def does_satisfay(y_1, y_2, y_3):
    min_dif = 0.01
    return abs(math.pow(y_1, 2) + math.pow(y_2, 2) - 1) > min_dif

def calculate_constraint_violation(model, x_test):
    y_pred = model.predict(x_test)
    num_violations = 0
    for item in y_pred:
        if not does_satisfay(item[0], item[1], item[2]):
            num_violations += 1
    return num_violations

'''

sample x1 from Gaussian

x: x(t), y(t), z(t) | (t): z
'''    
def create_synthetic_dataset_main(data_size=10000):
    count = 0
    t = 0
    x_data = []
    y_data = []
    x_data.append((math.sin(0), math.cos(0), 0))
    while count < data_size:
        # sample x, y, z
        x_item = math.sin(t) 
        y_item = math.cos(t)
        z_item = t
        y_data.append((x_item, y_item, z_item))
        x_data.append((x_item, y_item, z_item))
        t = t + 0.1
        count = count + 1
    
    
    return (x_data[:int(data_size/2)], y_data[:int(data_size/2)]), (x_data[int(data_size/2):int(3*data_size/4)], y_data[int(data_size/2):int(3*data_size/4)]), (x_data[int(3*data_size/4):data_size], y_data[int(3*data_size/4):data_size])
    

def train_secondary_network():
    (x_train_em, y_train_em), (x_val, y_val), (x_test_em, y_test_em) = create_synthetic_dataset_em()
    batch_size = 32
    maxlen = batch_size * 500
    variables = 3
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

def train_main_network(constraint_encoder):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = create_synthetic_dataset_main()
    # step 1: create embeddings
    # create training dataset for the constraint
    # Y1 < Y2 sample real valued uniformly distributed in -100 to 100
    ffn_model = u.create_FFN([3, 20, 3])
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
#    model_em = None
    train_main_network(constraint_encoder)