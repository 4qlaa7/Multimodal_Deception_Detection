import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video

import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video
from torchvision.models.video import r3d_18
import torch.nn.functional as F
from torchvision.transforms import Resize, Normalize, Compose

from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image


def extract_video_features(video_path, max_frames=100, frame_skip=5):
    video, _, _ = read_video(video_path, pts_unit='sec')
    video = video[:min(max_frames, len(video)), :, :, :].to(torch.float)
    print(f'Using read_video: Input video shape: {video.shape}')
    feature_extractor = r3d_18(pretrained=True)
    feature_extractor.fc = nn.Identity()  # Remove the last fully connected layer
    
    # Define the batch size for processing frames
    batch_size = 2
    
    with torch.no_grad():
        transform = Compose([
            Resize((32, 32)),  # Further reduce resolution
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :]),
            Normalize(mean=[0.43216, 0.394666, 0.37645], 
                      std=[0.22803, 0.22145, 0.216989])
        ])

        features_list = []

        # Process video frames batch-wise with more aggressive frame skipping
        for i in range(0, len(video), batch_size * frame_skip):
            batch_frames = video[i:i + batch_size * frame_skip:frame_skip]
            batch_frames = [transform(frame) for frame in batch_frames]
            batch_frames = torch.stack(batch_frames, dim=0)
            batch_frames = batch_frames.permute(1, 0, 2, 3)
            batch_frames = batch_frames.unsqueeze(0)

            # Extract features directly without concatenating
            features_batch = feature_extractor(batch_frames)

            # Append the features to the list
            features_list.append(features_batch)

            # Break the loop if the maximum number of frames is reached
            if len(features_list) * batch_size >= max_frames:
                break

    # Concatenate features from all batches along dimension 1
    features = torch.cat(features_list, dim=1)

    return features   
    
class DeceptionDatasetold(Dataset):
    def __init__(self, file_paths, labels, transform=None):

        self.file_paths = file_paths
        self.labels = labels
        print("started the features extraction")
        self.all_features = []
        for video_path in self.file_paths:
            self.all_features.append(extract_video_features(video_path))

        print("finished the features extraction")
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        #video_path = self.all_videos[idx]
        #features = extract_video_features(video_path)
        features = self.all_features[idx]
        label = self.labels[idx]

        #if self.transform:
        #    features = self.transform(features)

        return features, label

class DeceptionDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, features_cache_dir='features_cache2'):

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
        # Create the features_cache directory if it does not exist
        os.makedirs(features_cache_dir, exist_ok=True)
        
        print("Started the features extraction")
        self.all_features = []
        for video_path in self.file_paths:
            # Create a cache file path for the current video
            videoname_path = video_path.split(".")
            cache_file = os.path.join(features_cache_dir, os.path.basename(videoname_path[0]) + '.pickle')
            
            # Check if the cache file exists and load features if it does
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
            else:
                features = extract_video_features(video_path)
                # Save features to the cache file
                with open(cache_file, 'wb') as f:
                    pickle.dump(features, f)
            
            self.all_features.append(features)

        print("Finished the features extraction")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        #video_path = self.all_videos[idx]
        #features = extract_video_features(video_path)
        features = self.all_features[idx]
        label = self.labels[idx]

        #if self.transform:
        #    features = self.transform(features)

        return features, label


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        #print('forward x.shape: ', x.shape)
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=0.15)
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, p=0.15)
        x = self.fc3(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc

def create_dataset_and_loader(deception_folder, truthful_folder, test_size=0.2, batch_size=10):
    deception_files  = [os.path.join(deception_folder, f) for f in os.listdir(deception_folder)]
    truthful_files = [os.path.join(truthful_folder, f) for f in os.listdir(truthful_folder)]


    file_paths = deception_files + truthful_files
    labels = [0] * len(deception_files) + [1] * len(truthful_files)
    #print(labels) 
    #exit(0)


    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=test_size, stratify=labels)

    #train_dataset = AudioFeaturesDataset(train_files, train_labels)
    #test_dataset = AudioFeaturesDataset(test_files, test_labels)
    
    #vgg16 = load_vgg16()
    # Define VGG model
    
    train_dataset = DeceptionDataset(train_files, train_labels)
    test_dataset = DeceptionDataset(test_files, test_labels)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    


    return train_loader, test_loader

if __name__ == '__main__':
    

    # Set up data transforms
    transform = Compose([
                          Resize((256, 256)),  # resize frames to 256x256
                          transforms.ToTensor(),
                          Normalize(mean=[0.5, 0.5, 0.5], 
                                    std=[0.5, 0.5, 0.5])  # normalize pixel values
                        ])


    # Set up training and validation data loaders
    #train_dataset = DeceptionDataset(deceptive_dir=deception, truthful_dir=truthful, transform=transform)
    #train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    print("We are creating the dataset and dataloader")
    #deception_folder = "/home/leticia/datasets/video_temp/Clips/Deceptive"
    #truthful_folder = "/home/leticia/datasets/video_temp/Clips/Truthful"
    

    #deception_folder = "/home/leticia/datasets/Real-life_Deception_Detection_2016/Clips/Deceptive"
    #truthful_folder = "/home/leticia/datasets/Real-life_Deception_Detection_2016/Clips/Truthful"


    ##second dataset:
    deception_folder = "D:/EDU/Graduation Project/Project/Real-life_Deception_Detection_2016/Clips/Deceptive"
    truthful_folder = "D:/EDU/Graduation Project/Project/Real-life_Deception_Detection_2016/Clips/Truthful"

    train_loader, val_loader = create_dataset_and_loader(deception_folder, truthful_folder)
    print("We are done creating the dataset and dataloader")

    #val_dataset = DeceptionDataset(deceptive_dir=deception, truthful_dir=truthful, transform=transform)
    #val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)

    # Set up the model, loss function, and optimizer
    model = SimpleClassifier(input_size=512, num_classes=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train and evaluate the model for a fixed number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
