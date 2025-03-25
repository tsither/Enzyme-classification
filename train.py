import torch
import matplotlib.pyplot as plt
def train_and_evaluate(model, train_dataset, validation_dataset, optimizer, criterion, 
                      baseline=False, epochs=100, patience=10, min=0.001):
    print('Training model...')

    #### Training #####
    model.train()
    train_losses = []       #track training data loss across all epochs
    val_losses = []         #track validation set loss across all epochs
    val_accuracies = []     #track validation set accuracies across all epochs
    train_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    early_stop = False

    for e in range(epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {e}")
            break
            
        epoch_loss = 0
        for graph in train_dataset:
            optimizer.zero_grad()
            
            if baseline:
                out = model(graph.x)  #not training in batches
            else:
                out = model(graph.x, graph.edge_index)  #not training in batches
            
            out = out.unsqueeze(0)  #add dimension, pytorch expects a batch dimension, even if not using batches :(

            loss = criterion(out, graph.y.view(-1)) #compute loss
            epoch_loss += loss.item()           #add to epoch loss tracker
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        correct_val = 0 
        correct_train = 0
        val_epoch_loss = 0

        with torch.no_grad():  
            for graph in validation_dataset:    #accuracy score on validation set

                if baseline:
                    out = model(graph.x)  #not training in batches, dont need batch or edge index for MLP, not using graph structure
                else:
                    out = model(graph.x, graph.edge_index)  #not training in batches
                
                out = out.unsqueeze(0) 

                val_loss = criterion(out, graph.y.view(-1))  # Compute validation loss
                val_epoch_loss += val_loss.item()

                pred = out.argmax(dim=1)

                correct_val += int((pred == graph.y.view(-1)))

            for graph_train in train_dataset:  #accuracy score on train set

                if baseline:
                    out = model(graph_train.x)  #not training in batches, dont need batch or edge index for MLP, not using graph structure
                else:
                    out = model(graph_train.x, graph_train.edge_index)  #not training in batches
                
                out = out.unsqueeze(0) 

                pred = out.argmax(dim=1)

                correct_train += int((pred == graph_train.y.view(-1)))

        avg_val_loss = val_epoch_loss / len(validation_dataset)  # Average validation loss
        val_losses.append(avg_val_loss)

        val_accuracy = correct_val / len(validation_dataset)
        train_accuracy = correct_train / len(train_dataset)

        val_accuracies.append(val_accuracy)
        train_accuracies.append(train_accuracy)

        if e % 10 == 0:     #track epochs for key updates during training
            print(f"epoch: {e} ; train_loss: {avg_train_loss:.4f}, val_loss:{avg_val_loss:.4f}; val_acc:{val_accuracy:.4f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - min:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                early_stop = True

                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                print(f"Early stopping triggered. Best validation loss: {best_val_loss:.4f}")

        model.train()
        
    # If training completes without early stopping, ensure we use the best model
    if not early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return train_losses, val_losses, val_accuracies, train_accuracies, e


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