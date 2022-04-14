"""Train ResNet18 with CIFAR10"""
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from resnet import ResNet18


def get_arguments():

    parser = argparse.ArgumentParser(description="Train a ResNet-18 model on CIFAR10 using PyTorch")

    parser.add_argument("-ndev", "--num-devices", type=int, required=True, help="specify number of gpus")
    parser.add_argument("-dp", "--datapath", default="data", help="specify the path to the dataset folder")
    parser.add_argument("-e", "--epochs", required=True, type=int, help="specify the number of epochs to train for")
    parser.add_argument("-d", "--device", default="gpu", choices=["gpu", "cpu"], required=False, help="specify the compute device: gpu or cpu")
    parser.add_argument("-w", "--workers", type=int, default=2, help="specify the number of workers for the data loader")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.1, help="specify the learning rate for optimizers")
    parser.add_argument("-wd", "--weight-decay", type=float, default=5e-4, help="specify the weight decay value")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="specify the gamma value for optimizers that use momentum")
    parser.add_argument("-bs", "--batch-size", type=int, default=32, help="specify the batch size for training and testing")

    return parser.parse_args()


def transform_data(data):

    if data == "train":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    elif data == "test":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    return transform


def get_data(path):

    # download training data
    train_data = torchvision.datasets.CIFAR10(
        root=path,
        download=True,
        train=True,
        transform=transform_data("train"),
    )
    test_data = torchvision.datasets.CIFAR10(
        root=path,
        download=True,
        train=False,
        transform=transform_data("test"),
    )

    return train_data, test_data


def main():

    # get arguments from command line
    args = get_arguments()

    # download data
    train_data, _ = get_data(args.datapath)

    # define train
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_devices * args.batch_size, shuffle=True, num_workers=args.workers)

    ####################################################################################################################
    # ARGUENT HANDLING AND HYPERPARAMETER DIFINITIONS
    ####################################################################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18()

    if args.num_devices == 1:
        model = model.to(device)
    elif args.num_devices == 2:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model = model.to(device)
    elif args.num_devices == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.to(device)

    # get optimizer arguments
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)

    # selecting the loss function
    loss_fn = nn.CrossEntropyLoss()

    ####################################################################################################################
    # TIER DEFINITIONS
    ####################################################################################################################

    EPOCH_DATA_LOADING_TIME = [0 for _ in range(args.epochs)]
    EPOCH_TRAINING_TIME = [0 for _ in range(args.epochs)]
    EPOCH_ACCURACY = [0 for _ in range(args.epochs)]
    EPOCH_LOSS = [0 for _ in range(args.epochs)]
    TOTAL_RUNNING_TIME = [0 for _ in range(args.epochs)]

    ####################################################################################################################
    # TRAINING LOOP
    ####################################################################################################################

    print("Started training using SGD optimizer.")

    for epoch in range(args.epochs):

        # start the totalrunning time counter
        start_running_time_timer = time.perf_counter()

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # start the epoch data loading time timer
        start_epoch_data_loading_timer = time.perf_counter()

        for inputs, targets in train_data_loader:

            # end the epoch data loading time timer
            end_epoch_data_loading_timer = time.perf_counter()

            # store time to load data in each epoch
            EPOCH_DATA_LOADING_TIME[epoch] += end_epoch_data_loading_timer - start_epoch_data_loading_timer

            # start the epoch training time timer
            start_epoch_training_timer = time.perf_counter()

            # data movement, compute predictions, loss & gradients
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # end the epoch training time timer
            end_epoch_training_timer = time.perf_counter()

            # store time to train in each epoch
            EPOCH_TRAINING_TIME[epoch] += end_epoch_training_timer - start_epoch_training_timer

            # compute loss and accuracy per epch
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            start_epoch_data_loading_timer = time.perf_counter()

        current_accuracy = 100.0 * correct / total
        current_loss = train_loss / (len(train_data_loader) + 1)

        # end the total training time counter
        end_running_time_timer = time.perf_counter()

        # store the total running time for each epoch
        TOTAL_RUNNING_TIME[epoch] += end_running_time_timer - start_running_time_timer

        # add loss and accuracy of each epoch to an array
        EPOCH_ACCURACY[epoch] = current_accuracy
        EPOCH_LOSS[epoch] = current_loss

        # print training loss and training accuracy for each epoch.
        print(f"Epoch: {epoch + 1} | Training Loss: {current_loss: .3f} | Training Accuracy: {current_accuracy: .3f}")

    print("Finished training using SGD optimizer.")

    # compute required time values
    print(f"Training Time: {EPOCH_TRAINING_TIME[-1]}(s)")
    print(f"Total Time: {EPOCH_DATA_LOADING_TIME[-1] + EPOCH_TRAINING_TIME[-1]}(s)")


if __name__ == "__main__":

    main()
