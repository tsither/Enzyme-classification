import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv


class baselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        """
        Simple MLP to be used as a baseline model. Node feature vectors are averaged together, passed through MLP
        """
        super(baselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, node_embeddings):
        graph_embedding = torch.mean(node_embeddings, dim=0)         # Average node embeddings together to get a 'graph' representation that will be passed through MLP
        
        x = self.fc1(graph_embedding)       
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden, output_dim, num_layers, dropout, layer_norm):
        """
        Graph Convolutional Network with variable N layers (hyperparameter).
        - Convolutional layers use normalized degree matrix and Adjacency matrix self loops
        - Mean pooling into hidden dimension
        - optional layer norm
        """
        super(GCN, self).__init__()
        
        self.dropout = dropout  #dropout rate
        self.layer_norm = layer_norm    #boolean of whether to include layer normalization or not
        self.LN = nn.LayerNorm(hidden)
        self.num_layers = num_layers

        self.conv1 = GCNConv(input_dim, hidden, normalize=True)    #first GCN layer, needs to contain input dim

        self.layers = nn.ModuleList()       #initialize list to store N layers
        self.layers.append(self.conv1)     #always all first GCN layer to list

        for l in range(num_layers - 1):     #build out the number of layers according to the hyperparameter
            self.layers.append(GCNConv(hidden, hidden, normalize=True) )

        self.linear = Linear(hidden, output_dim)   #linear layer for classification on output dim


    def forward(self, x, edge_index):

        for i in range(self.num_layers):        #iterate through number of layers in model, hyperparameter
            x = self.layers[i](x, edge_index)   
            if self.layer_norm:
                x = self.LN(x)
            x = F.relu(x)

        x = torch.mean(x,dim=0) 

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        
        #softmax is automatically applied in nn.CrossEntropyLoss
        
        return x



class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, layer_norm, num_heads):
        """
        Graph Attention Network with variable N layers (hyperparameter).
        - Mean pooling into hidden dimension
        - optional layer norm
        """
        super(GAT, self).__init__()
        self.dropout = dropout  #dropout rate
        self.layer_norm = layer_norm    #boolean of whether to include layer normalization or not
        self.LN = nn.LayerNorm(hidden_dim)  #actual layer norm application
        self.num_layers = num_layers

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=False)   #concat=False: average output from different attention heads PRO:reduce complexity, dims, overfitting?

        self.layers = nn.ModuleList()       #initialize list to store N layers
        self.layers.append(self.gat1)     #always all first GAT layer to list

        for l in range(num_layers - 1):     #build out the number of layers according to the hyperparameter
            self.layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False) )

        self.linear = Linear(hidden_dim, output_dim)   #linear layer for classification on output dim


    def forward(self, x, edge_index):
        # Apply GAT layers
        for i in range(self.num_layers):        #iterate through number of layers in model, hyperparameter
            x = self.layers[i](x, edge_index)   
            if self.layer_norm:
                x = self.LN(x)
            x = F.relu(x)

        x = torch.mean(x,dim=0)     #average node embeddings to get a 'graph representation' vector

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x




