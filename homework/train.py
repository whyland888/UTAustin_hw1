from models import ClassificationLoss, model_factory, save_model
from utils import accuracy, load_data
import torch.optim as optim
import torch


def train(args):

    # Hyperparamaters
    batch_size = 128
    learning_rate = .001
    num_epochs = 9

    # Data loading
    train_loader = load_data(r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw1/data/train")
    valid_loader = load_data(r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw1/data/valid")

    model = model_factory[args.model]()
    criterion = ClassificationLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
