import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dim
        self.num_classes = output_dim
        self.fc = nn.Linear(self.input_dimension, self.num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)
