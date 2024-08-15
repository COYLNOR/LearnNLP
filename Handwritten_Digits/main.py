import json
import os
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt

from model import DenseRecognizer, ConvRecognizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_mnist(dir_path):
    transform = torchvision.transforms.Compose(
        [
            # convert to grayscale
            torchvision.transforms.Grayscale(num_output_channels=1),
            # convert to tensor
            torchvision.transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=dir_path, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dir_path, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def train(loader, model, optimizer, criterion, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        for i, (data, target) in enumerate(tqdm.tqdm(loader, desc=f"Epoch: {epoch}")):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            # forward pass
            # input: (N, C, H, W), target: (N)
            # output: (N, C)
            output = model(data)
            # cross entropy loss, from https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # input: (N, C), target: (N)
            # output: scalar
            loss = criterion(output, target)
            # backpropagation
            loss.backward()
            # update weights
            optimizer.step()
            loss_sum += loss.item()
        losses.append(loss_sum / len(loader))
        print(f"Epoch: {epoch}, Loss: {loss_sum / len(loader)}")

    # save model
    torch.save(model.state_dict(), "model.pth")

    # save loss plot
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig(os.path.join("output", "loss.png"))


def evaluate(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # get the index of the max log-probability
            # input: (N, C), output: (N)
            # the C in the project is 10 (0-9), and the argmax function returns the index of the maximum value
            # thus, the prediction is the digit with the highest probability
            prediction = torch.argmax(output, dim=1)
            correct += torch.sum(prediction == target).item()
            total += target.size(0)

    print(f"Accuracy: {correct / total}")


def infer(loader, model):
    data, target = next(iter(loader))
    torchvision.utils.save_image(data, "image.png")
    data = data.to(DEVICE)
    prediction = model(data)
    prediction = torch.argmax(prediction, dim=1)
    print(f"Prediction: {prediction}")
    print(f"Target: {target}")


def load_model(model, model_path="model.pth"):
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        if config == None:
            print("config.json is empty.")
            return

    data_dir = config.get("data", "./data")
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 1e-3)
    mode = config.get("mode", "default")
    model_type = config.get("model", "dense")

    train_dataset, test_dataset = read_mnist(data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    if model_type == "dense":
        model = DenseRecognizer().to(DEVICE)
    elif model_type == "conv":
        model = ConvRecognizer().to(DEVICE)
    else:
        print("Invalid model type.")
        return

    model_path = "model.pth"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    if mode == "default":
        train(train_loader, model, optimizer, criterion, epochs)
        evaluate(test_loader, model)
        infer(test_loader, model)
    elif mode == "train":
        train(train_loader, model, optimizer, criterion, epochs)
        evaluate(test_loader, model)
    elif mode == "inference":
        model = load_model(model, model_path)
        if model:
            infer(test_loader, model)
    elif mode == "evaluate":
        model = load_model(model, model_path)
        if model:
            evaluate(test_loader, model)
    else:
        print("Invalid mode.")


if __name__ == "__main__":
    main()
