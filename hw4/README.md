# How to run the code?

These are the options.

Train a ResNet-18 model on CIFAR10 using PyTorch

```
optional arguments:
  -h, --help            show this help message and exit
  -ndev NUM_DEVICES, --num-devices NUM_DEVICES
                        specify number of gpus
  -dp DATAPATH, --datapath DATAPATH
                        specify the path to the dataset folder
  -e EPOCHS, --epochs EPOCHS
                        specify the number of epochs to train for
  -d {gpu,cpu}, --device {gpu,cpu}
                        specify the compute device: gpu or cpu
  -w WORKERS, --workers WORKERS
                        specify the number of workers for the data loader
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        specify the learning rate for optimizers
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        specify the weight decay value
  -m MOMENTUM, --momentum MOMENTUM
                        specify the gamma value for optimizers that use momentum
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        specify the batch size for training and testing
```