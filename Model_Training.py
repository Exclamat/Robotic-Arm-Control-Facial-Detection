import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision.utils

try:
    from inception_resnet_v1 import InceptionResnetV1
except ImportError:
    print("="*50)
    print("ERROR: Could not import InceptionResnetV1.")
    print("Please make sure 'inception_resnet_v1.py' is in the same folder as this script.")
    print("="*50)
    exit()

DATA_DIR = "cropped_faces"
PRETRAINED_MODEL_PATH = "casia-webface.pt" 
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
VALID_SPLIT = 0.2
RANDOM_SEED = 42
MISCLASSIFIED_DIR = "misclassified_images" 


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class ConvertPilToRawTensor:
    def __call__(self, pil_img):
        return torch.tensor(np.array(pil_img), dtype=torch.float32).permute(2, 0, 1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    data_transform = transforms.Compose([
        transforms.Resize((160, 160)), 
        ConvertPilToRawTensor(),       
        transforms.Lambda(fixed_image_standardization)
    ])

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please make sure you have run the face_cropper.py script first.")
        return

    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transform)
    print(f"Dataset loaded: {len(dataset)} images found.")
    
    num_classes = len(dataset.classes)
    if num_classes == 0:
        print(f"Error: No classes (subfolders) found in '{DATA_DIR}'.")
        return
    print(f"Found {num_classes} classes (persons): {dataset.classes}")
    class_names = dataset.classes 

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALID_SPLIT * dataset_size))
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    print(f"Training with {len(train_indices)} images, validating with {len(val_indices)} images.")

    os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)
    print(f"Saving misclassified images to: {MISCLASSIFIED_DIR}")

    resnet = InceptionResnetV1(
        classify=True,
        num_classes=num_classes
    ).to(device)

    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Error: Pretrained model '{PRETRAINED_MODEL_PATH}' not found.")
        print(f"Please download it and place it in the same directory as this script.")
        return
        
    print(f"Loading pretrained weights from: {PRETRAINED_MODEL_PATH}")
    
    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True)
    
    if 'logits.weight' in state_dict:
        del state_dict['logits.weight']
    if 'logits.bias' in state_dict:
        del state_dict['logits.bias']

    resnet.load_state_dict(state_dict, strict=False) 
    print("Pretrained weights loaded successfully (final layer ignored).")

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(resnet.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10])

    print("\n--- Starting Training ---")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        resnet.train() 
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        train_loss = running_loss / len(train_indices)
        train_acc = running_corrects.double() / len(train_indices)
        
        resnet.eval() 
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad(): 
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                misclassified_mask = (preds != labels.data)
                if misclassified_mask.any():
                    misclassified_indices = torch.where(misclassified_mask)[0]
                    
                    for i in misclassified_indices:
                        img_tensor = inputs[i]
                        
                        img_unstd = (img_tensor.cpu() * 128.0 + 127.5) / 255.0
                        img_unstd = torch.clamp(img_unstd, 0, 1) 
                        
                        pred_name = class_names[preds[i]]
                        actual_name = class_names[labels.data[i]]
                        
                        filename = f"epoch_{epoch + 1}_pred_{pred_name}_actual_{actual_name}_batch_{batch_idx}_img_{i.item()}.jpg"
                        save_path = os.path.join(MISCLASSIFIED_DIR, filename)
                        
                        torchvision.utils.save_image(img_unstd, save_path)
                
        val_loss = running_loss / len(val_indices)
        val_acc = running_corrects.double() / len(val_indices)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Valid Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]}")

    print("\n--- Training Complete ---")
    
    save_path = "face_recognizer_model.pt"
    torch.save(resnet.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()