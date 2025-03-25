"""
By default contains option to train model on optimal hyperparameters.
"""

from torch_geometric.datasets import TUDataset
import torch
from train import train_and_evaluate, visualize
from models import baselineMLP, GCN, GAT
from data_processing import process_dataset


def main():

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

    input_dim = dataset.num_features

    # 80 - 10 - 10 split into train, validation, and test, returns tuple of (x,y)
    train_data, validation_data, test_data = process_dataset(dataset=dataset, random_seed=10, train_size=480, val_test_size=60)  

    model = GCN(input_dim=input_dim, hidden=128, output_dim=6, num_layers=4, dropout=0.5, layer_norm=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()     #automatically calculates log_softmax within loss function!

    train_losses, val_losses, acc, training_accuracies = train_and_evaluate(model,train_data,validation_dataset=validation_data, optimizer=optimizer,criterion=criterion, baseline=False, epochs=25)
    
    torch.save(model.state_dict(), 'saved_models/retrained_GCN_model.pth')

    print(f"final acc: {acc[-1]}")



    def evaluate(model,test_dataset,baseline=False):

        model.eval()
        correct = 0 

        with torch.no_grad():  
            for graph in test_dataset:

                out = model(graph.x, graph.edge_index)  #not training in batches
                
                out = out.unsqueeze(0)  #but pretend 

                pred = out.argmax(dim=1)

                correct += int((pred == graph.y.view(-1)))

        accuracy = correct / len(test_dataset)

        return accuracy

    print(f"GCN train: {evaluate(model, test_dataset=train_data)}")
    print(f"GCN val: {evaluate(model, test_dataset=validation_data)}")

    model = GCN()
    model.load_state_dict(torch.load("saved_models/best_GCN_model.pth"))
    model.eval()

    print(f"GCN val: {evaluate(model, test_dataset=validation_data)}")


if __name__ == '__main__':
    main()