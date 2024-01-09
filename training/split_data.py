import json
import random

def split_data(input_file, train_file, val_file, train_ratio=0.8):
    # Read the data from the input file
    with open(input_file, 'r') as file:
        data = [json.loads(line) for line in file]

    # Shuffle the data randomly
    random.shuffle(data)

    # Calculate the number of training samples
    train_size = int(len(data) * train_ratio)

    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Write the training data
    with open(train_file, 'w') as file:
        for entry in train_data:
            file.write(json.dumps(entry) + '\n')

    # Write the validation data
    with open(val_file, 'w') as file:
        for entry in val_data:
            file.write(json.dumps(entry) + '\n')

# File paths
input_file_path = 'data/ner_data_augmented.jsonl'  # Replace with your file path
train_file_path = 'data/train.jsonl'
val_file_path = 'data/val.jsonl'

# Call the function to split the data
split_data(input_file_path, train_file_path, val_file_path)
