import torch
import torch.nn.functional as F


class HighwayMLP(torch.nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=F.relu,
                 gate_activation=F.softmax):
        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = torch.nn.Linear(input_size, input_size)

        self.gate_layer = torch.nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):
        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)
