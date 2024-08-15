import torch


class DenseRecognizer(torch.nn.Module):
    """
    use dense layers to recognize handwritten digits
    """

    def __init__(self):
        """
        Linear layer from https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        The shape of input through the network is from (N, In) to (N, Out)
        Thus, the output shape of the input is (N, 10) on the last layer
        Shape: (N, 28 * 28) -> (N, 512) -> (N, 256) -> (N, 10)
        """

        super(DenseRecognizer, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    def forward(self, x):
        # flatten the input tensor to (batch_size, 28 * 28)
        x = x.view(-1, 28 * 28)
        x = self.net(x)
        return x


class ConvRecognizer(torch.nn.Module):
    """
    use convolutional layers to recognize handwritten digits
    """

    def __init__(self):
        """
        Conv2d layer from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        MaxPool2d layer from https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        the input shape is (N, 1, 28, 28)
        when the input goes through the first Conv2d layer, the output shape is (N, 32, 28, 28)
        due to the C_in is 1 and C_out is 32, 1 -> 32 at the second dimension
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 28
        W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 28
        when the output goes through the first MaxPool2d layer, the output shape is (N, 32, 14, 14)
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 14
        W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 14
        then the shape:
        first max pool (N, 32, 14, 14) ->
        second conv2d (N, 64, 14, 14) ->
        second max pool (N, 64, 7, 7) ->
        flatten (N, 64 * 7 * 7) ->
        first linear (N, 64 * 7 * 7) -> (N, 128)
        the output shape is (N, 10)
        """

        super(ConvRecognizer, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
