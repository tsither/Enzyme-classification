"""
Training Graph Attention Network using random search on 15 searches.
Hyperparameters:
- N epochs
- learning rate
- N GATConv layers
- Include layer normalization
- Size of hidden dim
- Dropout rate
- weight decay strength
- number of attention heads
"""

import torch
from enzyme_classification.train import train_and_evaluate
from enzyme_classification.models import GAT
from enzyme_classification.data_processing import process_dataset, INPUT_DIM, OUTPUT_DIM
import random
import json

GAT_hyperparameters = { "epochs" : [10, 30, 50],
                       "layers" : [2,3,4],
                       "layer norm" : [True],
                    "lr": [0.0001, 0.001, 0.01],
                   "hidden": [64, 128, 256],
                   "dropout": [0.3],
                   "weight_decay": [0.0001, 0.001, 0.01],
                   "num_heads": [2,3,4]}

def main():

    train_dataset, val_dataset, test_dataset = process_dataset()
    
    best_model = None
    best_accuracy = 0  # Initialize to track the best accuracy
    num_random_searches = 73  # Specify how many random combinations to sample
    best_hyperparameters = {}
    results = []

    for i in range(num_random_searches):
        print(f"N experiments: {i+1}")

        epochs = random.choice(GAT_hyperparameters["epochs"])
        layers = random.choice(GAT_hyperparameters["layers"])
        layer_norm = random.choice(GAT_hyperparameters["layer norm"])
        lr = random.choice(GAT_hyperparameters["lr"])
        hidden_dim = random.choice(GAT_hyperparameters["hidden"])
        dropout = random.choice(GAT_hyperparameters["dropout"])
        weight_decay = random.choice(GAT_hyperparameters["weight_decay"])
        num_heads = random.choice(GAT_hyperparameters["num_heads"])


        print(f"Training with epochs: {epochs}, lr: {lr}, hidden_dim: {hidden_dim}, dropout: {dropout}, weight_decay: {weight_decay}")

        model = GAT(INPUT_DIM, hidden_dim, OUTPUT_DIM, num_layers=layers, dropout=dropout, layer_norm=layer_norm, num_heads=num_heads)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()     #automatically calculates log_softmax within loss function!

        train_losses, val_losses, val_accuracies, train_accuracies, stopped_epoch = train_and_evaluate(model, train_dataset, validation_dataset=val_dataset, optimizer=optimizer, criterion=criterion, baseline=False, epochs=epochs)
        final_acc = val_accuracies[-1]
        final_train_acc = train_accuracies[-1]

        print(f"epoch: {stopped_epoch}")
        print(f"Final train ccc: {final_train_acc}")
        print(f"Final Val Accuracy: {final_acc}")

        results.append({
            'epochs': stopped_epoch,
            'lr': lr,
            'layers': layers,
            'layer norm': layer_norm,
            'hidden': hidden_dim,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'num heads' : num_heads,
            'train loss arc': train_losses,
            'val loss arc': val_losses,
            'val accuracies' : val_accuracies,
            'train accuracies': train_accuracies,
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
            best_hyperparameters["num_heads"] = num_heads

        print("--------- \n")
    
    print(f'Best epochs: {best_hyperparameters["epochs"]}') 
    print(f'Best N layers: {best_hyperparameters["layers"]}') 
    print(f'Layer norm: {best_hyperparameters["layer norm"]}') 
    print(f'Best lr: {best_hyperparameters["lr"]}')
    print(f'Best hidden dim size: {best_hyperparameters["hidden"]}')
    print(f'Best dropout rate: {best_hyperparameters["dropout"]}')
    print(f'Best weight decay: {best_hyperparameters["weight_decay"]}')
    print(f'Best N heads: {best_hyperparameters["num_heads"]}')


    with open('enzyme_classification/results/GAT.json', 'w') as f:
        json.dump(results, f, indent=4)


    print("Results saved to results/GAT.json")
    torch.save(best_model.state_dict(), 'enzyme_classification/saved_models/best_GAT_model.pth')

if __name__ == '__main__':
    main()