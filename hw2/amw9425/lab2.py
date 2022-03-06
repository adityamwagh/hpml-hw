"""Train CIFAR10 with PyTorch."""
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary

from resnet import ResNet18
from resnet_nobn import ResNet18_NoBN

if __name__ == "__main__":
    ####################################################################################################################
    # PROVISION FOR ARGUMENTS
    ####################################################################################################################

    parser = argparse.ArgumentParser(
        description="Train a ResNet-18 model on CIFAR10 using PyTorch")

    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        choices=["gpu", "cpu"],
        required=True,
        help=
        "specify the compute device: gpu or cpu. If there's no GPU, specifying gpu will default to cpu. ",
    )
    parser.add_argument("-dp",
                        "--datapath",
                        default="data",
                        required=True,
                        help="specify the path to dataset.")
    parser.add_argument("-e",
                        "--epochs",
                        required=True,
                        type=int,
                        help="specify the number of epochs to train for.")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        help="specify the number of workers for the data loader.",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="sgd",
        choices=["sgd", "sgdnest", "adam", "adagrad", "adadelta"],
        required=True,
        help="specify the optimizer for training.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.1,
        required=True,
        help="specify the learning rate for optimizers.",
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=5e-4,
        help="specify the weight decay value.",
    )  # ??
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=0.9,
        help="specify the gamma value for optimizers that use momentum.",
    )
    parser.add_argument(
        "-dbn",
        "--disable-batchnorm",
        action="store_true",
        help="specify whether to use batch normalization layers.",
    )

    parser.add_argument(
        "-q",
        "--question",
        choices={"c2", "c3", "c4", "c5", "c6", "c7"},
        required=True,
        help="specify which question to solve",
    )
    args = parser.parse_args()
    print(
        "=============================================================================================================="
    )
    print(f"Running the computations for Question: {args.question.upper()}")

    ####################################################################################################################
    # DATA LOADING
    ####################################################################################################################


    def transform_data(data):
        if data == "train":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010)),
            ])
        elif data == "test":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010)),
            ])

        return transform

    # download training data
    train_data = torchvision.datasets.CIFAR10(
        root=args.datapath,
        download=True,
        train=True,
        transform=transform_data("train"),
    )
    test_data = torchvision.datasets.CIFAR10(
        root=args.datapath,
        download=True,
        train=False,
        transform=transform_data("test"),
    )

    # define dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=args.workers)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=100,
                                                   shuffle=False,
                                                   num_workers=args.workers)

    ####################################################################################################################
    # ARGUMENT HANDLING AND HYPERPARAMETER DIFINITIONS
    ####################################################################################################################
    device = None

    # set the device for computation
    if args.device == "gpu":
        if torch.cuda.is_available():
            print(
                f"GPU is available. Training on {torch.cuda.get_device_name()}"
            )
            device = "cuda"
    else:
        device = "cpu"

    # choose model according to batchnorm specification
    model = ResNet18_NoBN() if args.disable_batchnorm else ResNet18()
    model = model.to(device)

    # get optimizer from the arguments
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )

    elif args.optimizer == "sgdnest":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)

    # cross entropy loss
    criterion = nn.CrossEntropyLoss()

    ####################################################################################################################
    # TIMER DEFINITIONS
    ####################################################################################################################
    EPOCH_DATA_LOADING_TIME = [0 for _ in range(args.epochs)]
    EPOCH_TRAINING_TIME = [0 for _ in range(args.epochs)]
    EPOCH_ACCURACY = [0 for _ in range(args.epochs)]
    EPOCH_LOSS = [0 for _ in range(args.epochs)]
    TOTAL_RUNNING_TIME = [0 for _ in range(args.epochs)]

    ####################################################################################################################
    # TRAINING LOOP
    ####################################################################################################################

    print(f"Started training using {args.optimizer.upper()} optimizer.")

    for epoch in range(args.epochs):

        # start the totalrunning time counter
        start_running_time_timer = time.perf_counter()

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # start the epoch data loading time timer
        start_epoch_data_loading_timer = time.perf_counter()

        for batch_idx, (inputs, targets) in enumerate(train_data_loader):

            # end the epoch data loading time timer
            end_epoch_data_loading_timer = time.perf_counter()

            # store time to load data in each epoch
            EPOCH_DATA_LOADING_TIME[epoch] += (end_epoch_data_loading_timer -
                                               start_epoch_data_loading_timer)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # start the epoch training time timer
            start_epoch_training_timer = time.perf_counter()

            # compute predictions, loss & gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # perform gradient descent
            optimizer.step()

            # end the epoch training time timer
            end_epoch_training_timer = time.perf_counter()

            # store time to train in each epoch
            EPOCH_TRAINING_TIME[epoch] += (end_epoch_training_timer -
                                           start_epoch_training_timer)

            # compute loss and accuracy per epch
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            start_epoch_data_loading_timer = time.perf_counter()

        current_accuracy = 100.0 * correct / total
        current_loss = train_loss / (len(train_data_loader) + 1)
        # add loss and accuracy of each epoch to an array
        EPOCH_ACCURACY.append(current_accuracy)
        EPOCH_LOSS.append(current_loss)

        print(
            f"Epoch: {epoch + 1} | Training Loss: {current_loss: .3f} | Training Accuracy: {current_accuracy: .3f}"
        )

        # end the total training time counter
        end_running_time_timer = time.perf_counter()
        TOTAL_RUNNING_TIME[
            epoch] += end_running_time_timer - start_running_time_timer

    print(f"Finished training using {args.optimizer} optimizer.")

    ####################################################################################################################
    # QUESTION WISE COMPUTATIONS
    ####################################################################################################################

    if args.question == "c2":
        # print data loading time and training timme for each epoch
        for i, (a, b, c) in enumerate(
                zip(EPOCH_DATA_LOADING_TIME, EPOCH_TRAINING_TIME,
                    TOTAL_RUNNING_TIME)):
            print(
                f"Epoch {i+1} | Data Loading Time: {a: .3f} sec | Training Time: {b: .3f} sec | Running Time: {c: .3f} sec"
            )
    elif args.question in ["c3", "c4"]:
        print(
            f"Total Data loading time for {args.workers} workers: {sum(EPOCH_DATA_LOADING_TIME)} sec"
        )
    elif args.question == "c5":
        print(
            f"Average running time for {args.epochs} epochs on a {args.device.upper()}: {sum(TOTAL_RUNNING_TIME) / args.epochs} sec"
        )
    elif args.question == "c6":
        print(
            f"Average training time per epoch for {args.optimizer.upper()}: {sum(EPOCH_TRAINING_TIME) / args.epochs} sec"
        )
        print(
            f"Average loss per epoch for {args.optimizer.upper()}: {sum(EPOCH_LOSS) / args.epochs}"
        )
        print(
            f"Average top-1 training per epoch for {args.optimizer.upper()}: {sum(EPOCH_ACCURACY) / args.epochs}"
        )
    elif args.question == "c7":
        print(
            f"Average training loss per epoch without batch normalization layers: {sum(EPOCH_LOSS) / args.epochs}"
        )
        print(
            f"Average top-1 training accuracy per epoch without batch normalization layers: {sum(EPOCH_ACCURACY) / args.epochs}"
        )
    ####################################################################################################################
    # MODEL PARAMETERS
    ####################################################################################################################

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(summary(model, (3, 32, 32)))
print("\n")