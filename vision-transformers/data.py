import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import time

# Get logger
logger = logging.getLogger('vision_transformer.data')

def prepare_data(batch_size=4, num_worker=2):
    """
    Prepare CIFAR10 datasets and dataloaders
    """
    logger.info(f"Preparing data with batch_size={batch_size}, num_worker={num_worker}")
    
    # Define transformations
    logger.info("Creating data transformations")
    train_tranform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root='data_CIFAR10', train=True, download=False, transform=train_tranform)
    logger.info("Training dataset loaded successfully")
    logger.info(f"Training dataset size: {len(trainset)}")
    
    # Create training dataloader
    logger.info("Creating training dataloader")
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_worker
    )
    logger.info(f"Training dataloader created with {len(trainloader)} batches")

    # Define test transformations
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    testset = torchvision.datasets.CIFAR10(root='data_CIFAR10', train=False, download=False, transform=test_transform)
    logger.info("Test dataset loaded successfully")
    logger.info(f"Test dataset size: {len(testset)}")

    # Create test dataloader
    logger.info("Creating test dataloader")
    testloader = torch.utils.data.DataLoader(
        dataset=testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_worker
    )
    logger.info(f"Test dataloader created with {len(testloader)} batches")

    # Define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    logger.info(f"Classes: {classes}")

    return trainloader, testloader, classes