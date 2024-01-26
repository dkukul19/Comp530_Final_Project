import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import json
import os
from PIL import Image
import sys
import argparse
sys.stdout.flush()
VERBOSE = False
class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, transform=None):
        self.data = []
        with open(jsonl_file, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_img_path = self.data[idx]['img']  #'img/42953.png'

        _, filename = os.path.split(json_img_path)
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.data[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Hateful Memes dataset')
    parser.add_argument('--train_jsonl', type=str, default='train_backdoored.jsonl', help='Path to the training JSONL file')
    parser.add_argument('--validation_jsonl', type=str, default='dev_backdoored.jsonl', help='Path to the validation JSONL file')
    parser.add_argument('--img_dir', type=str, default='img_backdoored', help='Directory where images are stored')
    parser.add_argument('--model_save_path', type=str, default='hateful_memes_model_resnet_40by40_20percent_bottomright.pth', help='Path to save the trained model')
    args = parser.parse_args()

#no need to resize here since the input is already resized to 224,224
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HatefulMemesDataset(jsonl_file=args.train_jsonl, img_dir=args.img_dir, transform=transform)
    validation_dataset = HatefulMemesDataset(jsonl_file=args.validation_jsonl, img_dir=args.img_dir,#"backdoored_datasets/seth",#
                                             transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 25
    total_batches = len(train_loader)
    percent_to_report = 1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if VERBOSE:
                if (i + 1) % (total_batches // (100 // percent_to_report)) == 0:
                    percent_complete = (i + 1) / total_batches * 100
                    print(f"Epoch {epoch + 1}/{num_epochs}, {percent_complete:.0f}% complete, Current Loss: {loss.item()}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / total_batches}")






# TESTING

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                ## Print predictions for each image
                #for idx, (pred, true_label) in enumerate(zip(predicted, labels)):
                #    image_path = validation_dataset.data[idx]['img']
                #    print(f"Image: {image_path}, Predicted: {pred.item()}, Actual: {true_label.item()}")

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy}%", flush=True)

    #torch.save(model.state_dict(), 'hateful_memes_model_resnet_40by40_20percent_bottomright.pth')
    torch.save(model.state_dict(), args.model_save_path)