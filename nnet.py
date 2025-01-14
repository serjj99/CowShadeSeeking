import torch
import torch.nn as nn

from torchsummary import summary


class NNet(nn.Module):
    
    def __init__(self, in_dim, out_dim,
                 h_layers, fun_act = None, out_act = None,
                 device = 'cpu', dropout = None):
        """
        Neural Network model.

        Args:
            in_dim (int): Number of input dimensions.
            out_dim (int): Number of output dimensions.
            h_layers (list): List of hidden layer sizes.
            fun_act (str, optional): Activation function for hidden layers. Defaults to None.
            out_act (str, optional): Activation function for output layer. Defaults to None.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.
            dropout (float, optional): Dropout probability for regularization. Defaults to None.
        """

        
        super(NNet, self).__init__()

        self.input_size = in_dim
        self.hidden_layers = h_layers
        self.input_layer = nn.Linear(in_dim, h_layers[0])
        self.hidden = nn.ModuleList(
            [nn.Linear(h_layers[i], h_layers[i+1]) for i in range(len(h_layers) - 1)]
        )
        self.output_layer = nn.Linear(h_layers[-1], out_dim)
        self.dropout = dropout

        if fun_act == 'relu':
            self.fun_act = torch.relu
        elif fun_act == 'tanh':
            self.fun_act = torch.tanh
        else:
            self.fun_act = None

        if out_act == 'relu':
            self.out_act = torch.relu
        else:
            self.out_act = None
        
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(p = self.dropout)
        
        self.to(device)

    def forward(self, x):

        x = self.fun_act(self.input_layer(x))

        for layer in self.hidden:
        
            x = layer(x)
            if self.fun_act is not None:
                x = self.fun_act(x)
                
            if self.dropout is not None:
                x = self.dropout_layer(x)

        x = self.output_layer(x)
        if self.out_act is not None:
            x = self.out_act(x)

        return x
    
    def print_model(self):
        
        print(self.eval())
        summary(self, (1, self.input_size))
        
    def load_model(self, file_path):

        self.load_state_dict(torch.load(file_path))