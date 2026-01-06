import os
import time
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg13

# =====================
# Emotion labels mapping
# =====================
emotion_table = {
    'neutral': 0,
    'happiness': 1,
    'surprise': 2,
    'sadness': 3,
    'anger': 4,
    'disgust': 5,
    'fear': 6,
    'contempt': 7
}

num_classes = len(emotion_table)

# =====================
# Dataset wrapper for FER+
# =====================
from torch.utils.data import Dataset
from PIL import Image
import csv

class FERPlusDataset(Dataset):
    def __init__(self, base_folder, folders, csv_file, transform=None, training_mode="majority"):
        self.samples = []
        self.transform = transform
        self.training_mode = training_mode

        for folder in folders:
            folder_path = os.path.join(base_folder, folder)
            with open(os.path.join(folder_path, csv_file), 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    img_path = os.path.join(folder_path, row[0])
                    labels = np.array(list(map(int, row[2:])), dtype=np.float32)

                    if training_mode == "majority":
                        label = np.argmax(labels)
                        self.samples.append((img_path, label))
                    elif training_mode == "probability" or training_mode == "crossentropy":
                        label = labels / np.sum(labels)
                        self.samples.append((img_path, label))
                    elif training_mode == "multi_target":
                        label = (labels > 0).astype(np.float32)
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.long)
        return img, label

# =====================
# Model builder
# =====================
def build_model(model_name="VGG13", num_classes=8):
    if model_name == "VGG13":
        model = vgg13(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # grayscale input
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

# =====================
# Loss function selection
# =====================
def get_loss(training_mode):
    if training_mode in ["majority", "crossentropy"]:
        return nn.CrossEntropyLoss()
    elif training_mode == "probability":
        return nn.KLDivLoss(reduction="batchmean")
    elif training_mode == "multi_target":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")

# =====================
# Training loop
# =====================
def train(base_folder, training_mode="majority", model_name="VGG13", max_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output folders
    output_model_path = os.path.join(base_folder, "models")
    output_model_folder = os.path.join(output_model_path, f"{model_name}_{training_mode}")
    os.makedirs(output_model_folder, exist_ok=True)

    logging.basicConfig(filename=os.path.join(output_model_folder, "train.log"),
                        filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(f"Starting with training mode {training_mode} using {model_name} model and max epochs {max_epochs}.")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = FERPlusDataset(base_folder, ["FER2013Train"], "label.csv", transform, training_mode)
    val_dataset   = FERPlusDataset(base_folder, ["FER2013Valid"], "label.csv", transform, "majority")
    test_dataset  = FERPlusDataset(base_folder, ["FER2013Test"], "label.csv", transform, "majority")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    model = build_model(model_name, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = get_loss(training_mode)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            if training_mode in ["majority", "crossentropy"]:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            elif training_mode == "probability":
                log_probs = torch.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, labels)
                _, predicted = torch.max(outputs, 1)
                _, targets = torch.max(labels, 1)
                correct += (predicted == targets).sum().item()
                total += labels.size(0)
            elif training_mode == "multi_target":
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels.bool()).all(dim=1).sum().item()
                total += labels.size(0)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_model_folder, f"model_{best_epoch}.pth"))

            test_acc = evaluate(model, test_loader, device)
            best_test_acc = max(best_test_acc, test_acc)

            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%, test_acc={test_acc*100:.2f}%")
        else:
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")

    logging.info("")
    logging.info(f"Best validation accuracy: {best_val_acc*100:.2f}%, epoch {best_epoch}")
    logging.info(f"Best test accuracy: {best_test_acc*100:.2f}%")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--base_folder", type=str, required=True,
                        help="Base folder containing FER2013Train, FER2013Valid, FER2013Test with label.csv")
    parser.add_argument("-m", "--training_mode", type=str, default='majority',
                        help="Training mode: majority, probability, crossentropy, multi_target")
    parser.add_argument("-e", "--max_epochs", type=int, default=100)
    args = parser.parse_args()

    train(args.base_folder, args.training_mode, max_epochs=args.max_epochs)