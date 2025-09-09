import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from utils import load_experiment

def parse_args():
    parser = argparse.ArgumentParser(description='Test ViT model on random CIFAR10 images')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--checkpoint', type=str, default='model_final.pt', help='Checkpoint file name')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to test')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from experiment: {args.exp_name}")
    config, model, _, _, _ = load_experiment(args.exp_name, checkpoint_name=args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load CIFAR10 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='data_CIFAR10', train=False, download=False, transform=transform)
    
    # Select random images
    num_images = min(args.num_images, len(testset))
    indices = random.sample(range(len(testset)), num_images)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if num_images >= 10 else plt.subplots(1, num_images, figsize=(15, 3))
    axes = axes.flatten() if num_images >= 10 else axes
    
    # Process each image
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            # Get image and label
            image, label = testset[idx]
            true_label = classes[label]
            
            # Convert image for display
            img_display = np.transpose(image.numpy(), (1, 2, 0))
            img_display = img_display * 0.5 + 0.5  # Denormalize
            
            # Get prediction
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            outputs, _ = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = classes[predicted.item()]
            
            # Display image with labels
            axes[i].imshow(img_display)
            color = 'green' if predicted.item() == label else 'red'
            axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}", color=color)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    print(f"Visualization saved to model_predictions.png")
    plt.show()

if __name__ == "__main__":
    main()