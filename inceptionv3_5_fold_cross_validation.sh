#!/path/to/your/python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from PIL import Image


n_epochs = 50
learning_rate = 0.03
weight_decay = 1e-4
momentum = 0.9
batch_size = 64
size_iv3 = 299
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label


data_dir = '/path/to/your/data/set'
all_image_paths = []
all_labels = []
label_map = {'class_0': 0, 'class_1': 1}

for class_name in os.listdir(data_dir):
    if class_name in label_map:
        class_dir = os.path.join(data_dir, class_name)
        label = label_map[class_name]
        for image_name in os.listdir(class_dir):
            all_image_paths.append(os.path.join(class_dir, image_name))
            all_labels.append(label)

all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(all_image_paths, all_labels)):
    print(f"--- Fold {fold+1}/5 ---")
    
    train_paths, val_paths = all_image_paths[train_index], all_image_paths[val_index]
    train_labels, val_labels = all_labels[train_index], all_labels[val_index]

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = inception_v3(weights='Inception_V3_Weights.DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_fold_accuracy = 0.0
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        epoch_val_correct = 0
        epoch_val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                epoch_val_total += labels.size(0)
                epoch_val_correct += (predicted == labels).sum().item()
        
        epoch_accuracy = epoch_val_correct / epoch_val_total
        
        if epoch_accuracy > best_fold_accuracy:
            best_fold_accuracy = epoch_accuracy
            model_save_path = f'inception_v3_fold_{fold+1}_best.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"  > Saved model for fold {fold+1} with accuracy {best_fold_accuracy:.4f} .")

    
    model.eval()
    fold_preds = []
    fold_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fold_preds.extend(probabilities.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())

    np.save(f'fold_{fold+1}_preds.npy', np.array(fold_preds))
    np.save(f'fold_{fold+1}_labels.npy', np.array(fold_labels))
    print(f"Validation results for fold {fold+1} have been saved to fold_{fold+1}_preds.npy and fold_{fold+1}_labels.npy。")

print(" All 5-fold cross-validation complete.")