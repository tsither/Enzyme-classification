"""
Training MLP for use as a baseline model for GNN graph classification task; using random search on 15 searches.
Hyperparameters:
- N epochs
- learning rate
- Size of hidden dim
- Dropout rate
- weight decay strength
"""

import torch
from enzyme_classification.train import train_and_evaluate
from enzyme_classification.models import baselineMLP
from enzyme_classification.data_processing import process_dataset, INPUT_DIM, OUTPUT_DIM
import random
import json


baseline_hyperparameters = { "epochs" : [10, 30, 50],
                    "lr": [0.0001, 0.001, 0.01],
                   "hidden": [64, 128, 256],
                   "dropout": [0.3],
                   "weight_decay": [0.001, 0.01]}

def main():

    train_dataset, val_dataset, test_dataset = process_dataset()
    
    best_model = None
    best_accuracy = 0  # Initialize to track the best accuracy
    num_random_searches = 15  # Specify how many random combinations to sample
    best_hyperparameters = {}
    results = []

    for i in range(num_random_searches):

        epochs = random.choice(baseline_hyperparameters["epochs"])
        lr = random.choice(baseline_hyperparameters["lr"])
        hidden_dim = random.choice(baseline_hyperparameters["hidden"])
        dropout = random.choice(baseline_hyperparameters["dropout"])
        weight_decay = random.choice(baseline_hyperparameters["weight_decay"])

        print(f"Training with epochs: {epochs}, lr: {lr}, hidden_dim: {hidden_dim}, dropout: {dropout}, weight_decay: {weight_decay}")

        model = baselineMLP(INPUT_DIM, hidden_dim=hidden_dim, output_dim=OUTPUT_DIM, dropout=dropout)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()     #automatically calculates log_softmax within loss function!
        
        train_losses, val_losses, val_accuracies, train_accuracies, stopped_epoch = train_and_evaluate(model, train_dataset, validation_dataset=val_dataset, optimizer=optimizer, criterion=criterion, baseline=True, epochs=epochs)
        final_val_acc = val_accuracies[-1]
        final_train_acc = train_accuracies[-1]

        print(f"Epochs: {stopped_epoch}")
        print(f"Final Training data Accuracy: {final_train_acc}")
        print(f"Final Validation Accuracy: {final_val_acc}")

        results.append({                #store results of all experiments in dist, later write to JSON 
            'epochs': epochs,
            'stopped epoch': stopped_epoch,
            'lr': lr,
            'hidden': hidden_dim,
            'weight_decay': weight_decay,
            'train loss arc': train_losses,
            'val loss arc': val_losses,
            'val accuracies' : val_accuracies,
            'Final Val Accuracy': final_val_acc,
            
        })

        if final_val_acc > best_accuracy:
            best_accuracy = final_val_acc
            print("** New best model ** ")

            best_model = model
            best_hyperparameters["epochs"] = stopped_epoch
            best_hyperparameters["lr"] = lr
            best_hyperparameters["hidden"] = hidden_dim
            best_hyperparameters["weight_decay"] = weight_decay

        print("--------- \n")
    
    print(f'Best epochs: {best_hyperparameters["epochs"]}') 
    print(f'Best lr: {best_hyperparameters["lr"]}')
    print(f'Best hidden dim size: {best_hyperparameters["hidden"]}')
    print(f'Best weight decay: {best_hyperparameters["weight_decay"]}')


    with open('enzyme_classification/results/MLP.json', 'w') as f:
        json.dump(results, f, indent=4)


    print("Results saved to results/MLP.json")
    torch.save(best_model.state_dict(), 'enzyme_classification/saved_models/best_MLP_model.pth')

if __name__ == '__main__':
    main()