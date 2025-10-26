import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import copy
import random
from collections import defaultdict

# ============================================================================
#  VGG6 MODEL DEFINITION
# ============================================================================
class VGG6(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(VGG6, self).__init__()   
        # Block 1: 64 filters
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # Block 2: 128 filters
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 3: 256 filters
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)       
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )        
        self._initialize_weights() 
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)       
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x) 
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
# ============================================================================
# DATA LOADING
# ============================================================================
def get_cifar10_dataloaders(batch_size=128, num_workers=2, data_dir='./data'):   
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    # augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])   
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])   
    print("Loading CIFAR-10 dataset...")   
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )  
    # Split training set into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    _, val_subset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )   
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True) 
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']   
    print(f"Training: {len(train_subset)}, Validation: {len(val_subset)}, Test: {len(test_dataset)}")    
    return train_loader, val_loader, test_loader, classes

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()       
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()    
    return running_loss / len(val_loader), 100. * correct / total
def test_model(model, test_loader, device, classes):
    """Test model and return per-class accuracies"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1   
    accuracy = 100. * correct / total
    class_accuracies = {}   
    print(f'\nTest Accuracy: {accuracy:.2f}%\nPer-class:')
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[classes[i]] = class_acc
            print(f'{classes[i]}: {class_acc:.2f}%')
    
    return accuracy, class_accuracies
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.01,
                weight_decay=1e-4, device='cuda', save_path='best_model.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    model = model.to(device)
    history = defaultdict(list)
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\nTraining on {device} | Epochs: {num_epochs} | LR: {learning_rate}")
    print("-" )
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')  
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device) 
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            print(f" New best model saved! Val Acc: {val_acc:.2f}%")      
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)      
        print(f'Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | '
              f'Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | '
              f'LR={current_lr:.6f} | Time={time.time()-epoch_start:.2f}s')
    print(f"\n Training completed in {(time.time()-start_time)/60:.2f} min")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    return history, best_model_state

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(history, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))   
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)   
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)   
    # Learning rate
    ax3.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('LR')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)  
    # Val accuracy with best point
    ax4.plot(epochs, history['val_acc'], 'r-', linewidth=2)
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax4.plot(best_epoch, best_acc, 'go', markersize=10, 
            label=f'Best: {best_acc:.2f}% (Epoch {best_epoch})')
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved training curves")
    plt.close()

def plot_class_accuracies(class_accuracies, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())  
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='navy', linewidth=1.2) 
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')   
    plt.title('Per-Class Test Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y') 
    avg_acc = np.mean(accuracies)
    plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_acc:.1f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_accuracies.png'), dpi=300, bbox_inches='tight')
    print(f" Saved class accuracies")
    plt.close()

# ============================================================================
# Randomizaton Cuda check and model save
# ============================================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(history, test_accuracy, class_accuracies, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, 'results_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("VGG6 CIFAR-10 TRAINING RESULTS\n")
        f.write("=" + "\n\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n\n")
        best_epoch = np.argmax(history['val_acc']) + 1
        best_val_acc = max(history['val_acc'])
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})\n\n")
        f.write("Per-Class Accuracies:\n")
        for cls, acc in class_accuracies.items():
            f.write(f"  {cls:12}: {acc:6.2f}%\n")
    
    print(f" Results saved to: {os.path.join(save_path, 'results_summary.txt')}")

# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    print("VGG6 ON CIFAR-10")    
    set_seed(42)
    # Configuration
    config = {
        'batch_size': 128,
        'num_epochs': 100,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
        'num_workers': 2,
        'save_path': 'vgg6_results'
    } 
    os.makedirs(config['save_path'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")   
    # Load data
    print("DATA LOADING")
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        data_dir=os.path.join(config['save_path'], 'data')
    )
    # Create model
    print("\n" )
    print("MODEL")
    print("\n" )
    model = VGG6(num_classes=10, dropout_rate=config['dropout_rate'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VGG6 created | Parameters: {total_params:,}")  
    # Train
    print("\n")
    print("TRAINING")
    print("=")
    history, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device,
        save_path=os.path.join(config['save_path'], 'best_model.pth')
    )
    # Test
    print("\n")
    print("TESTING")
    print("=")
    model.load_state_dict(best_model_state)
    model = model.to(device)
    test_accuracy, class_accuracies = test_model(model, test_loader, device, classes)
    
    # Save results and plot
    print("\n")
    print("SAVING RESULTS & PLOTS")
    print("=")
    save_results(history, test_accuracy, class_accuracies, config['save_path'])
    plot_training_curves(history, config['save_path'])
    plot_class_accuracies(class_accuracies, config['save_path'])
    
    print("\n")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Results saved to: {config['save_path']}/")
    print(f"Plots saved: training_curves.png, class_accuracies.png")
    print("=")
    
    return {
        'test_accuracy': test_accuracy,
        'class_accuracies': class_accuracies,
        'history': history,
        'model': model,
        'config': config
    }

if __name__ == "__main__":
        results = main()
