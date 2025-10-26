# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import argparse
import random
import numpy as np
import os
import hashlib
import json

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to generate unique hash for config dict
def get_config_hash(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

# VGG6 Architecture
class VGG6(nn.Module):
    def __init__(self, num_classes=10, activation='relu'):
        super(VGG6, self).__init__()
        self.activation = self._get_activation(activation)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU()
        }
        if activation in activations:
            return activations[activation]
        print(f"Warning: Unknown activation '{activation}', using ReLU")
        return nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_dataloaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def validate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(testloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        set_seed(42)
        
        # Check for Google Drive
        if os.path.exists('/content/gdrive/MyDrive'):
            base_path = '/content/gdrive/MyDrive/vgg6_models'
            print("✓ Using Google Drive for model storage")
        else:
            base_path = 'models'
            print("Using local storage for models")
        
        os.makedirs(base_path, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'✓ Using device: {device}')
        
        # Generate unique hash for this config
        config_dict = {
            'activation': config.activation,
            'optimizer': config.optimizer,
            'learning_rate': float(config.learning_rate),
            'batch_size': int(config.batch_size),
            'epochs': int(config.epochs)
        }
        config_hash = get_config_hash(config_dict)
        
        # Create unique filenames based on config hash
        checkpoint_path = os.path.join(base_path, f'checkpoint_{config_hash}.pth')
        best_model_path = os.path.join(base_path, f'best_model_{config_hash}.pth')
        
        print(f'\nConfig Hash: {config_hash}')
        print(f'Activation: {config.activation}, Optimizer: {config.optimizer}')
        print(f'LR: {config.learning_rate}, Batch: {config.batch_size}, Epochs: {config.epochs}\n')
        
        # Skip if already trained well
        val_acc_threshold = 80.0
        if os.path.isfile(best_model_path):
            try:
                best_checkpoint = torch.load(best_model_path, map_location=device)
                saved_val_acc = best_checkpoint.get('val_acc', 0)
                if saved_val_acc >= val_acc_threshold:
                    print(f'✓ Config already achieved {saved_val_acc:.2f}% >= {val_acc_threshold}%')
                    print(f'Skipping training.\n')
                    wandb.log({'val_acc': saved_val_acc, 'status': 'skipped'})
                    return
            except Exception as e:
                print(f'Could not read best model: {e}')
        
        trainloader, testloader = get_dataloaders(config.batch_size)
        model = VGG6(num_classes=10, activation=config.activation).to(device)
        print(f'✓ Model initialized with {config.activation} activation')
        
        criterion = nn.CrossEntropyLoss()
        
        # Select optimizer
        optimizers = {
            'sgd': lambda: optim.SGD(model.parameters(), lr=config.learning_rate,
                                    momentum=0.9, weight_decay=5e-4),
            'nesterov': lambda: optim.SGD(model.parameters(), lr=config.learning_rate,
                                         momentum=0.9, weight_decay=5e-4, nesterov=True),
            'adam': lambda: optim.Adam(model.parameters(), lr=config.learning_rate,
                                      weight_decay=5e-4),
            'rmsprop': lambda: optim.RMSprop(model.parameters(), lr=config.learning_rate,
                                            weight_decay=5e-4),
            'adagrad': lambda: optim.Adagrad(model.parameters(), lr=config.learning_rate,
                                            weight_decay=5e-4),
            'nadam': lambda: optim.NAdam(model.parameters(), lr=config.learning_rate,
                                        weight_decay=5e-4)
        }
        
        optimizer = optimizers.get(config.optimizer, optimizers['adam'])()
        print(f'✓ Using {config.optimizer} optimizer with lr={config.learning_rate}')
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        
        start_epoch = 0
        best_val_acc = 0.0
        
        # Resume from checkpoint if exists
        if os.path.isfile(checkpoint_path):
            print(f'\n✓ Loading checkpoint from: {checkpoint_path}')
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint['best_val_acc']
                print(f' Resumed from epoch {start_epoch}, Best val acc: {best_val_acc:.2f}%\n')
            except Exception as e:
                print(f'Error loading checkpoint: {e}')
                print('Starting fresh training...\n')
        else:
            print('No checkpoint found, starting fresh.\n')
        
        # Training loop
        try:
            for epoch in range(start_epoch, config.epochs):
                train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, testloader, criterion, device)
                
                scheduler.step()
                
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                print(f'Epoch [{epoch+1}/{config.epochs}] '
                      f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
                
                # Save checkpoint periodically or when improving
                should_save = ((epoch + 1) % 5 == 0) or (val_acc > best_val_acc)
                
                if should_save:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'config': config_dict
                    }, checkpoint_path)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_acc': val_acc,
                        'config': config_dict
                    }, best_model_path)
                    print(f'✓ New best model saved! Val Acc: {val_acc:.2f}%')
            
            print(f'\n{"="}')
            print(f' Training completed!')
            print(f'Best validation accuracy: {best_val_acc:.2f}%')
            print(f'Best model: {best_model_path}')
            print(f'{"="}\n')
            
            # Clean up checkpoint
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
                print(' Checkpoint cleaned up.\n')
                
        except KeyboardInterrupt:
            print(f'\nTraining interrupted.')
            print(f'Checkpoint saved at: {checkpoint_path}')
            print(f'Resume by running again.\n')
            raise
        except Exception as e:
            print(f'\nTraining error: {e}')
            print(f'Checkpoint saved at: {checkpoint_path}')
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG6 CIFAR-10 with W&B')
    parser.add_argument('--sweep', action='store_true', help='Run as W&B sweep')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'sigmoid', 'tanh', 'silu', 'gelu'])
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'nesterov', 'adam', 'rmsprop', 'adagrad', 'nadam'])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    if args.sweep:
        train()
    else:
        config = {
            'activation': args.activation,
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
        train(config)
