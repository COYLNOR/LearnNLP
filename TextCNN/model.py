import torch


class Conv1DTextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        """
        Conv1d layer from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        max pooling layer from https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html

        The input shape of the Conv1d layer is (N, L), where N is the batch size and L is the length of the input
        Shape:
        Embedding: (N, L, embed_size) ->
        First Conv1d: (N, 16, L-2) -> L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = L-2
        First MaxPool1d: (N, 16, L_out/3) -> L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = L_out/3
        Second Conv1d: (N, 32, L_out/3-2) -> L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = L_out/3-2
        Second MaxPool1d: (N, 32, L_out/9) -> L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = L_out/9
        Third MaxPool1d: (N, 32, 0) -> L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 0
        fc: (N, 32) -> (N, num_classes)
        """
        super(Conv1DTextCNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(embed_size, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(16, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
        )
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.fc(x)
        return x


class Conv2DTextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        """
        conv2d layer from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        max pooling layer from https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        Shape:
        Embedding: (N, L, embed_size) ->
        unsqueeze: (N, 1, L, embed_size) ->
        First Conv2d: (N, 16, L-2, 1) ->
            Hout = (Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1 = L-2
            Wout = (Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1 = 1
        First MaxPool2d: (N, 16, (L-2)/3, 1) ->
            Hout = (Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1 = Hin/3 = (L-2)/3
            Wout = (Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1 = 1
        Second Conv2d: (N, 32, (L-2)/9, 1) ->
            Hout = (Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1 = Hin-2 = (L-2)/3-2
            Wout = (Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1 = 1
        Second MaxPool2d: (N, 32, (L-2)/9-2/3, 1) ->
            Hout = (Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1 = Hin/3 = (L-2)/9-2/3
            Wout = (Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1 = 1
        squeeze: (N, 32, (L-6)/9) ->
        max_pool1d: (N, 32, 1) ->
            Lout = (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 1
        fc: (N, 32) -> (N, num_classes)
        """
        super(Conv2DTextCNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(3, embed_size)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 1)),
            torch.nn.Conv2d(16, 32, kernel_size=(3, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 1)),
        )
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.fc(x)
        return x


class TextCNN(torch.nn.Module):
    """
    from https://dennybritz.com/posts/wildml/understanding-convolutional-neural-networks-for-nlp/
    TextCNN has three Conv1d layers with different kernel sizes, 2, 3, and 4,
    the output of each Conv1d layer goes through a max pooling layer,
    and the output of the max pooling layer is concatenated together

    Shape:
    Input: (N, L) ->
    Embedding: (N, L, embed_size) ->
    permute: (N, embed_size, L) ->
    Conv1d: (N, channel_sizes[i], L-kernel_sizes[i]+1) ->
        Lout = (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = L-kernel_sizes[i]+1
    MaxPool1d: (N, channel_sizes[i], 1) ->
        Lout = (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 = 1
    Concat: (N, sum(channel_sizes), 1) ->
    Dropout: (N, sum(channel_sizes), 1) ->
    Flatten: (N, sum(channel_sizes)) ->
    fc: (N, num_classes)
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        num_classes,
        max_len=100,
        channel_sizes=[100, 100, 100],
        kernel_sizes=[2, 3, 4],
        dropout=0.5,
    ):
        super(TextCNN, self).__init__()
        stride = 1
        padding = 0

        self.embedding = torch.nn.Embedding(vocab_size, embed_size)

        self.conv_list = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    embed_size,
                    channel_sizes[i],
                    kernel_size=(kernel_sizes[i],),
                    stride=stride,
                    padding=padding,
                )
                for i in range(len(channel_sizes))
            ]
        )

        self.pool_list = torch.nn.ModuleList(
            [
                torch.nn.MaxPool1d(
                    kernel_size=(max_len - kernel_sizes[i] + 1),
                    stride=stride,
                    padding=padding,
                )
                for i in range(len(kernel_sizes))
            ]
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(sum(channel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        out = [torch.nn.functional.relu(conv(x)) for conv in self.conv_list]
        out = [pool(o) for o, pool in zip(out, self.pool_list)]
        out = torch.cat(out, dim=1)
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
