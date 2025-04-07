# library imports
import torch
import torchvision
import torchvision.transforms as transforms
import time
import copy
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def define_transforms():
    """
    Define transformations for training, validation, and test datasets.
    training data: resize to 256 * 256, center cropping, randomized horizontal & vertical flipping, and normalization
    validation and test data: resize to 256 * 256, center cropping, and normalization
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # data_transforms = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    return data_transforms

class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=10):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.data_dir, target_class)

            for video_folder in os.listdir(target_dir):
                video_path = os.path.join(target_dir, video_folder)
                if not os.path.isdir(video_path):
                    continue

                # Get all valid frames
                frames = []
                for f in os.listdir(video_path):
                    if f.lower().endswith(('.jpg', '.png')) and f.startswith('frame_'):
                        try:
                            frame_num = int(f.split('_')[1].split('.')[0])
                            frames.append(frame_num)
                        except (IndexError, ValueError):
                            continue

                frames = sorted(frames)

                # Only add samples if we have enough frames
                if len(frames) >= self.sequence_length:
                    # Add all possible sequences
                    for i in range(len(frames) - self.sequence_length + 1):
                        sample = (video_path, frames[i], class_index)
                        samples.append(sample)
                else:
                    print(f"Warning: Skipping {video_path} - only {len(frames)} frames available")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_frame, class_index = self.samples[idx]
        frames = []

        for i in range(self.sequence_length):
            frame_num = start_frame + i
            # Try both padded and non-padded formats
            for frame_format in [f"frame_{frame_num:04d}.jpg", f"frame_{frame_num}.jpg"]:
                frame_path = os.path.join(video_path, frame_format)
                if os.path.exists(frame_path):
                    try:
                        frame = Image.open(frame_path).convert('RGB')
                        if self.transform:
                            frame = self.transform(frame)
                        frames.append(frame)
                        break  # Found the frame, move to next
                    except Exception as e:
                        print(f"Error loading {frame_path}: {e}")
                        continue

        # If we couldn't load enough frames, return None
        if len(frames) < self.sequence_length:
            print(f"Warning: Only {len(frames)} frames loaded from {video_path}")
            return None

        frames = torch.stack(frames)
        return frames, class_index

def create_datasets(data_dir, train_perc, val_perc, test_perc, sequence_length=10):
    data_transforms = define_transforms()

    full_dataset = VideoFrameDataset(data_dir, transform=data_transforms['train'], sequence_length=sequence_length)

    train_size = int(train_perc * len(full_dataset))
    val_size = int(val_perc * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                             [train_size, val_size, test_size])

    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']

    class_names = full_dataset.classes
    num_classes = len(class_names)

    return train_dataset, val_dataset, test_dataset, class_names, num_classes

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers=2):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    return dataloaders, dataset_sizes

class ResNetLSTM(nn.Module):
    def __init__(self, resnet_model, lstm_hidden_size, lstm_num_layers, num_classes):
        super(ResNetLSTM, self).__init__()
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove the final FC layer
        self.lstm = nn.LSTM(
            input_size=2048 if isinstance(resnet_model, models.ResNet) else resnet_model.fc.in_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.resnet(x)
        x = x.view(batch_size, seq_length, -1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

def train_model(model, model_dir, criterion, optimizer, dataloaders, dataset_sizes, scheduler=None, device="cpu",
                num_epochs=1, plot_path=None):  # Add plot_path argument
    """
    Train the model using transfer learning

    Args:
        model (torchvision.models): model to train
        model_dir (str): path to directory to save model
        criterion (torch.nn.modules.loss): loss function
        optimizer (torch.optim): optimizer
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        dataset_sizes (dict): dictionary of sizes of training and validation sets
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        device (torch.device): device to train on
        num_epochs (int): number of epochs to train for
        plot_path (str): path to save the training/validation plots (optional)
    """
    # load the model to GPU if available
    model = model.to(device)
    since = time.time()

    # initialize best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize lists to store loss and accuracy values
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # loop over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate the data loader
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the weight gradients
                optimizer.zero_grad()

                # forward pass to get outputs and calculate loss and track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backpropagation to get the gradients with respect to each weight only if in train phase
                    if phase == 'train':
                        loss.backward()
                        # update the weights
                        optimizer.step()

                # convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * inputs.size(0)
                # track number of correct predictions
                running_corrects += torch.sum(preds == labels.data)

            # step along learning rate scheduler when in train
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()

            # calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))

            # Store loss and accuracy values
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            # if model performs better on val set, save weights as the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, model_dir)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {:3f}'.format(best_acc))

    # load the weights from best model
    model.load_state_dict(best_model_wts)

    # Plot the training and validation loss and accuracy
    plt.figure(figsize=(12, 5))

    # Convert accuracy histories to CPU numpy arrays
    train_acc_cpu = [acc.cpu().numpy() if torch.is_tensor(acc) else acc
                     for acc in train_acc_history]
    val_acc_cpu = [acc.cpu().numpy() if torch.is_tensor(acc) else acc
                   for acc in val_acc_history]

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_cpu, label='Train Accuracy')
    plt.plot(val_acc_cpu, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Save the plot to the specified path if provided
    if plot_path:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()  # Display the plot if no path is provided

    plt.close()  # Close the plot to free up memory

    return model

def test_model(model, dataloaders, device, class_names, plot_path=None):
    """
    Test the trained model performance on test dataset and generate confusion matrix

    Args:
        model: trained model
        dataloaders: dictionary of dataloaders
        device: device to run on
        class_names: list of class names
        plot_path: optional path to save confusion matrix plot
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if plot_path:
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {plot_path}")
    else:
        plt.show()

    plt.close()

    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return all_preds, all_labels, cm