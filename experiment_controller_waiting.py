import sys
import json

def convert_to_vocab(float_str):
    out = []
    for ch in float_str:
        if ch == '.':
            out.append(10)
        elif ch == '-':
            out.append(11)
        else:
            out.append(int(ch))
    if len(out) < 10:
        out.append(10)  # append .
        while len(out) < 10:
            out.append(0)   # keep appending 0
    return out

def create_synthetic_dataset_sec(y_train, maxlen=20000, input_dim=1, x_range=10):
    count = 0
    x_train = []
    y_train = []
    while count < maxlen:
        count += 1
        x_train_item = np.random.randint((0, 200), input_dim)
        x_train_item_v = [convert_to_vocab(str(x_train_item[0]))[:x_range], convert_to_vocab(str(x_train_item[1]))[:x_range]]
        x_train.append(x_train_item_v)
        if (x_train_item[0] < x_train_item[1]):
            y_train_item = 1
        else:
            y_train_item = 0
        y_train.append(y_train_item)
    # convert to vocabulary
    
    return (x_train[:int(maxlen/2)], y_train[:int(maxlen/2)]), (x_train[int(maxlen/2):], y_train[int(maxlen/2):])

def train_secondary_network(y_train):
    (x_train_em, y_train_em), (x_test_em, y_test_em) = create_synthetic_dataset_sec(y_train)
    pass

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
    
    pass

if __name__=="__main__":
    (x_train, y_train), (x_val, y_val) = create_dataset_main(sys.argv[1], sys.argv[2], sys.argv[3])
    model_em = train_secondary_network(y_train)
#    model_em = None
    train_main_network(x_train, y_train, x_val, y_val, model_em)