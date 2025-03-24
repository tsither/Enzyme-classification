"""
- Load dataset, process data into ML readable format, 
- split into train, val and test with random shuffling
- Standardize train, val and test with mean and std from training data
"""
from torch_geometric.datasets import TUDataset
import torch
import matplotlib.pyplot as plt


DATASET = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

INPUT_DIM = DATASET.num_node_features
OUTPUT_DIM = DATASET.num_classes

def process_dataset(dataset=DATASET, random_seed=10, train_size=480, val_test_size=60):

    DATASET = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

    torch.manual_seed(random_seed)

    indices = torch.randperm(len(dataset)).tolist()     #random shuffle dataset

    val_idx = train_size + val_test_size        #integer index to allow splitting for val and test sets

    train_dataset = [dataset[i] for i in indices[:train_size]]      #list of 480 graphs representing training data

    val_dataset = [dataset[i] for i in indices[train_size:val_idx]] #list of 60 graphs representing validation data

    test_dataset = [dataset[i] for i in indices[val_idx:]]          #list of 60 graphs for test data

    #### Standarize data using mean and std from TRAINING DATA, NEVER TOUCH THE TEST DATA OR YOU FAIL ####
    all_features = torch.cat([data.x for data in train_dataset], dim=0)

    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True)

    for graph_t in train_dataset:                   #train
        graph_t.x = (graph_t.x - mean) / std

    for graph_v in val_dataset:                     #validation set
        graph_v.x = (graph_v.x - mean) / std

    for graph_test in test_dataset:                 #test set
        graph_test.x = (graph_test.x - mean) / std

    return train_dataset, val_dataset, test_dataset



def visualize(train_losses, val_losses):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()