import torch
import tqdm
import matplotlib.pyplot as plt
import json

from dataset import read_imdb
from model import Conv1DTextCNN, Conv2DTextCNN, TextCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader(data_dir, batch_size):
    train_dataset, test_dataset, lang = read_imdb(data_dir)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, test_dataloader, lang


def train(train_dataloader, model, optimizer, criterion, epochs):
    model_path = "model.pth"
    loss_list = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for x, y in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        loss_list.append(loss_sum / len(train_dataloader))
        print(f"Epoch {epoch}, Loss: {loss_sum / len(train_dataloader)}")

    torch.save(model.state_dict(), model_path)

    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss.png")


def eval(test_dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy: {correct / total}")


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    model_type = config.get("model", "conv1d")
    embed_size = config.get("embedding_dim", 100)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 1e-3)
    epochs = config.get("num_epochs", 10)
    data_dir = config.get("data_dir", "data")

    train_dataloader, test_dataloader, lang = get_dataloader(data_dir, batch_size)

    if model_type == "conv1d":
        model = Conv1DTextCNN(
            vocab_size=lang.n_words, embed_size=embed_size, num_classes=2
        )
    elif model_type == "conv2d":
        model = Conv2DTextCNN(
            vocab_size=lang.n_words, embed_size=embed_size, num_classes=2
        )
    elif model_type == "textcnn":
        model = TextCNN(vocab_size=lang.n_words, embed_size=embed_size, num_classes=2)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train(train_dataloader, model, optimizer, criterion, epochs)
    eval(test_dataloader, model)


if __name__ == "__main__":
    main()
