import os
import time
import argparse
from model_file import ResNet50
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from logger import CSVLogger

from torch.optim.lr_scheduler import MultiStepLR

# Defining the Training Loop


def train_val_model(epochs, train_loader, val_loader, model, loss_fn, optimizer, scheduler):
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        """
        Model training
        """
        model.train()
        running_train_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:

            X, y = imgs.to(DEVICE), labels.to(DEVICE)
            train_pred = model(X)
            train_loss = loss_fn(train_pred, y)

            # Back prop

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
            _, predicted = train_pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(DEVICE)).sum().item()

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_loss_values.append(epoch_train_loss)
        epoch_train_accuracy = 100.0 * correct / total
        train_accuracy_values.append(epoch_train_accuracy)

        """
        Model Validation
        """
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in val_loader:

            X, y = imgs.to(DEVICE), labels.to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred, y)

            running_val_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(DEVICE)).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataloader)
        val_loss_values.append(epoch_val_loss)
        epoch_val_accuracy = 100.0 * correct / total
        val_accuracy_values.append(epoch_val_accuracy)
        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | val Loss: {epoch_val_loss:.3f} | val Accuracy: {epoch_val_accuracy} %")

        if epoch_val_accuracy > 85.0:
            end_time = time.time()

            print(f"Time taken: {end_time - start_time} seconds\n")
            # print time in minutes
            print(f"Time taken: {(end_time - start_time)/60} minutes\n")
            break

        row = {"epoch": str(epoch), "val_accuracy": str(epoch_val_accuracy), "val_loss": str(epoch_val_loss)}
        csv_logger.writerow(row)


if __name__ == "__main__":

    DATA_PATH = "./data"
    EPOCHS = 200
    Lr = 0.01

    if torch.cuda.is_available() == True:
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(DEVICE)

    loss = nn.CrossEntropyLoss()

    model = ResNet50()
    print(model)
    model.to(DEVICE)
    # model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=Lr, momentum=0.9, nesterov=True, weight_decay=5e-4)

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    """
    Data Related Stuff
    """
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_data = torchvision.datasets.CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
    val_data = torchvision.datasets.CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=2)

    csv_logger = CSVLogger(fieldnames=["epoch", "val_accuracy", "val_loss"])
    """
    Model Training And Evaluation
    """
    print("Started Training on", DEVICE, "\n")

    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []

    train_val_model(EPOCHS, train_dataloader, val_dataloader, model, loss, optimizer, scheduler)

    print("Finished Training\n")
    # save the training & valing loss & accuracy to the disk
    os.makedirs(os.path.join(os.getcwd(), "metrics"), exist_ok=True)
    ## save model weights
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "metrics", "model_weights1.pth"))
    np.save(os.path.join(os.getcwd(), "metrics", "train_loss_values.npy"), train_loss_values)
    np.save(os.path.join(os.getcwd(), "metrics", "val_loss_values.npy"), val_loss_values)
    np.save(os.path.join("metrics", "val_accuracy.npy"), val_accuracy_values)
    np.save(os.path.join("metrics", "train_accuracy.npy"), train_accuracy_values)
