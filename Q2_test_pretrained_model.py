# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import os

# VGG6 Model Definition
class VGG6(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG6, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.gelu(self.conv1_1(x))
        x = F.gelu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.gelu(self.conv2_1(x))
        x = F.gelu(self.conv2_2(x))
        x = self.pool2(x)
        
        x = F.gelu(self.conv3_1(x))
        x = F.gelu(self.conv3_2(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load test data
def get_test_loader(batch_size=100):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                            shuffle=False, num_workers=2)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return testloader, classes

# Test function
def test_model(model, testloader, device, classes):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    test_acc = 100. * correct / total
    
    print("\n" + "*****")
    print("TEST RESULTS")
    print("*****")
    print(f"Overall Test Accuracy: {test_acc:.2f}%")
    print(f"Correct: {correct}/{total}")
    print("\nPer-Class Accuracies:")
    print("******")
    
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"{classes[i]:12}: {class_acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
    
    print("*******")
    
    return test_acc

def main():
    parser = argparse.ArgumentParser(description='Test pre-trained VGG6 model')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to model weights file')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for testing')
    args = parser.parse_args()
    
    print("*****")
    print("TESTING PRE-TRAINED BEST MODEL")
    print("*********")
    print("\nBest Configuration (from sweep):")
    print("  Activation: GELU")
    print("  Optimizer: Adam")
    print("  Learning Rate: 0.001")
    print("  Batch Size: 64")
    print("  Epochs: 50")
    print("  W&B Run: northern-sweep-328")
    print("  Sweep ID: 92kzsowc")
    print("="*80)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        print("Please provide the correct path using --model_path")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Model path: {args.model_path}")
    
    # Load model
    print("\nLoading model...")
    model = VGG6().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully")
    
    # Load test data
    print("\nLoading CIFAR-10 test set...")
    testloader, classes = get_test_loader(batch_size=args.batch_size)
    print(f"Test set loaded ({len(testloader.dataset)} images)")
    
    # Test
    test_acc = test_model(model, testloader, device, classes)
    
    print(f"\n Final Test Accuracy: {test_acc:.2f}%")
    print(" Testing completed!")

if __name__ == "__main__":
    main()
