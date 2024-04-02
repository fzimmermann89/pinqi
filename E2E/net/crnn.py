import torch
from torch import nn, Tensor
from typing import Callable, Tuple


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2, bias=False)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2, bias=False)
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, hidden_iteration, hidden, bias=None):
        in_to_hidden = self.i2h(x)
        hidden_to_hidden = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)
        out = in_to_hidden + hidden_to_hidden + ih_to_ih
        if bias is not None:
            out = bias[None, :, None, None] + out
        out = self.activation(out)
        return out


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.CRNN_model = CRNNcell(input_size, hidden_size, kernel_size)
        self.bias_f = nn.Parameter(torch.zeros(hidden_size))
        self.bias_b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input, input_iteration):
        """
        Input Shape: n_seq, n_b, n_c (2), nx, ny
        Output Shape: n_seq, n_b, n_c (2), nx, ny
        """
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = torch.zeros(size_h, device=input.device)

        # forward
        hidden = hid_init
        output_f = []
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden, self.bias_f)
            output_f.append(hidden)
        output_f = torch.stack(output_f)

        # backward
        hidden = hid_init
        output_b = []
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i - 1], hidden, self.bias_b)
            output_b.append(hidden)
        output_b = torch.stack(output_b[::-1])

        output = output_f + output_b
        return output


class CRNNlayer(nn.Module):
    def __init__(self, filters: int, kernel_size: int):
        """
        CRNN Layer
        """
        super().__init__()
        self.conv_x = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size // 2, bias=True)
        self.conv_h = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, h):
        x = self.conv_x(x)
        h = self.conv_h(h)
        x = self.activation(x + h)
        return x


class CRNN(nn.Module):
    """
    Convolutional Recurrent Network for MRI Reconstruction
    """

    def __init__(self, filters: int = 64, kernel_size: int = 3, layers: int = 5):
        """
        Based on https://github.com/cq615/Deep-MRI-Reconstruction/blob/master/cascadenet_pytorch/model_pytorch.py
        filters: Number of filters in all layers
        kernel_size: CNN kernel size
        layers: Total number of layers in each iteration. Will do 1 BCRNN layer, (layers-2) CRNN layers, and 1 Conv Layer.
        """

        super().__init__()
        self.bcrnn = BCRNNlayer(2, filters, kernel_size)
        self.crnn = nn.ModuleList([CRNNlayer(filters, kernel_size) for i in range(layers - 2)])
        self.conv_out = nn.Conv2d(filters, 2, kernel_size, padding=kernel_size // 2)
        self.filters = filters

    def forward(self, x, DC: Tuple[Callable[...,Tensor]] = (lambda x: x,)):
        """
        DC: Tuple of Data Consistency Layers, lenth determines number of iterations
        """
        n_batch, n_seq, n_x, n_y = x.size()
        hidden_old = [torch.zeros(n_batch * n_seq, self.filters, n_x, n_y, device=x.device)] * (len(self.crnn))
        h0 = torch.zeros(n_seq, n_batch, self.filters, n_x, n_y, device=x.device)
        out = [x,]
        for dc in DC:
            real_x = torch.view_as_real(x).permute(1, 0, 4, 2, 3)
            hidden = []
            h0 = self.bcrnn(real_x, h0)
            z = h0.flatten(end_dim=1)
            for layer, h in zip(self.crnn, hidden_old):
                z = layer(z, h)
                hidden.append(z)
            hidden_old = hidden
            z = self.conv_out(z)
            z = torch.view_as_complex(torch.permute_copy(z.view(n_seq, n_batch, 2, n_x, n_y), (1, 0, 3, 4, 2)))
            x = x + z
            x = dc(x)
            out.append(x)
        return out
