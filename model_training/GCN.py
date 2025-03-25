"""
Training Graph Convolutional Network using random search on 15 searches.
Hyperparameters:
- N epochs
- learning rate
- N GCNConv layers
- Include layer normalization
- Size of hidden dim
- Dropout rate
- weight decay strength
"""

import torch
from ..train import train_and_evaluate
from ..models import GCN
from ..data_processing import process_dataset, INPUT_DIM, OUTPUT_DIM
import random
import json

GCN_hyperparameters = { "epochs" : [10, 30, 50],
                       "layers" : [2,3,4],
                       "layer norm" : [True],
                    "lr": [0.0001, 0.001, 0.01],
                   "hidden": [64, 128, 256],
                   "dropout": [0.3],
                   "weight_decay": [0.0001, 0.001, 0.01]}

def main():

    train_dataset, val_dataset, test_dataset = process_dataset()

    best_model = None
    best_accuracy = 0  # Initialize to track the best accuracy
    num_random_searches = 25  # Specify how many random combinations to sample
    best_hyperparameters = {}
    results = []

    for i in range(num_random_searches):
        print(f"N experiment: {i + 1}")

        epochs = random.choice(GCN_hyperparameters["epochs"])
        layers = random.choice(GCN_hyperparameters["layers"])
        layer_norm = random.choice(GCN_hyperparameters["layer norm"])
        lr = random.choice(GCN_hyperparameters["lr"])
        hidden_dim = random.choice(GCN_hyperparameters["hidden"])
        dropout = random.choice(GCN_hyperparameters["dropout"])
        weight_decay = random.choice(GCN_hyperparameters["weight_decay"])

        print(f"Training with epochs: {epochs}, lr: {lr}, hidden_dim: {hidden_dim}, dropout: {dropout}, weight_decay: {weight_decay}")

        model = GCN(INPUT_DIM, hidden_dim, OUTPUT_DIM, num_layers=layers, dropout=dropout, layer_norm=layer_norm)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()     #automatically calculates log_softmax within loss function!

        train_losses, val_losses, val_accuracies, train_accuracies, stopped_epoch = train_and_evaluate(model, train_dataset, validation_dataset=val_dataset, optimizer=optimizer, criterion=criterion, baseline=False, epochs=epochs)
        final_acc = val_accuracies[-1]
        final_train_acc = train_accuracies[-1]

        print(f"Epochs:{stopped_epoch}")
        print(f"Final Train Accuracy: {final_train_acc}")
        print(f"Final Val Accuracy: {final_acc}")

        results.append({                #store results of all experiments in dist, later write to JSON 
            'epochs': epochs,
            'stopped_epoch': stopped_epoch,
            'lr': lr,
            'layers': layers,
            'layer norm': layer_norm,
            'hidden': hidden_dim,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'train loss arc': train_losses,
            'val loss arc': val_losses,
            'val accuracies' : val_accuracies,
            'Final Accuracy': final_acc
        })

        if final_acc > best_accuracy:
            best_accuracy = final_acc
            print("** New best model ** ")
            best_model = model
            best_hyperparameters["epochs"] = stopped_epoch
            best_hyperparameters["layers"] = layers
            best_hyperparameters["layer norm"] = layer_norm
            best_hyperparameters["lr"] = lr
            best_hyperparameters["hidden"] = hidden_dim
            best_hyperparameters["dropout"] = dropout
            best_hyperparameters["weight_decay"] = weight_decay

        print("--------- \n")
    
    print(f'Best epochs: {best_hyperparameters["epochs"]}') 
    print(f'Best N layers: {best_hyperparameters["layers"]}') 
    print(f'Layer norm: {best_hyperparameters["layer norm"]}') 
    print(f'Best lr: {best_hyperparameters["lr"]}')
    print(f'Best hidden dim size: {best_hyperparameters["hidden"]}')
    print(f'Best dropout rate: {best_hyperparameters["dropout"]}')
    print(f'Best weight decay: {best_hyperparameters["weight_decay"]}')

    with open('enzyme_classification/results/GCN.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to results/GCN.json")

    print("Results saved to results/GCN.txt")
    torch.save(best_model.state_dict(), 'enzyme_classification/saved_models/best_GCN_model.pth')

if __name__ == '__main__':
    main()