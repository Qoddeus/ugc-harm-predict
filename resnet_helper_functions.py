# resnet_helper_functions.py

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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import random

def define_transforms():
    """
    Define transformations for training, validation, and test datasets.
    Enhanced with more augmentation techniques for training data.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),  # Random crop instead of center crop for training
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Color augmentation
            transforms.RandomRotation(10),  # Slight rotation
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

    return data_transforms

class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=16, stride=4, temporal_augment=True):
        """
        Enhanced video frame dataset with temporal augmentation and stride sampling

        Args:
            data_dir: Directory with class folders containing video frame folders
            transform: Image transforms to apply
            sequence_length: Number of frames per sequence
            stride: Step size when sampling frames (lower = more overlap)
            temporal_augment: Whether to apply temporal augmentation
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.stride = stride
        self.temporal_augment = temporal_augment
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
                    # Add sequences with stride for better coverage
                    for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
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

        # Temporal augmentation: randomly skip frames or change playback speed
        frame_indices = list(range(self.sequence_length))
        if self.temporal_augment and self.transform == 'train':
            # Random frame skipping (simulation of different speeds)
            if random.random() < 0.3:
                if random.random() < 0.5:  # Slow down
                    frame_indices = [min(i // 2, self.sequence_length - 1) for i in range(self.sequence_length)]
                else:  # Speed up
                    frame_indices = [min(i * 2, self.sequence_length - 1) for i in range(self.sequence_length)]
                frame_indices = sorted(set(frame_indices))
                # Pad if needed
                while len(frame_indices) < self.sequence_length:
                    frame_indices.append(frame_indices[-1])

        for i in frame_indices:
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
            # Pad with the last valid frame if we have at least one
            if frames:
                while len(frames) < self.sequence_length:
                    frames.append(frames[-1])
            else:
                return None

        frames = torch.stack(frames)
        return frames, class_index

def create_datasets(data_dir, train_perc, val_perc, test_perc, sequence_length=16, stride=4):
    data_transforms = define_transforms()

    full_dataset = VideoFrameDataset(data_dir, transform=data_transforms['train'],
                                     sequence_length=sequence_length, stride=stride,
                                     temporal_augment=True)

    # Calculate sizes
    dataset_size = len(full_dataset)
    train_size = int(train_perc * dataset_size)
    val_size = int(val_perc * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )

    # Override transforms
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']

    class_names = full_dataset.classes
    num_classes = len(class_names)

    return train_dataset, val_dataset, test_dataset, class_names, num_classes

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers=2):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    return dataloaders, dataset_sizes

class ResNetLSTM(nn.Module):
    def __init__(self, resnet_model, lstm_hidden_size, lstm_num_layers, num_classes, dropout_rate=0.5):
        super(ResNetLSTM, self).__init__()

        # Remove the final FC layer from ResNet
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-1])

        # Get feature size from ResNet
        if isinstance(resnet_model, models.ResNet):
            feature_size = 2048 if "resnet50" in str(resnet_model.__class__).lower() else resnet_model.fc.in_features
        else:
            feature_size = resnet_model.fc.in_features

        # Bidirectional LSTM for better temporal pattern capture
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional for better pattern capture
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classification head with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 256)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

        # Batch normalization
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()

        # CNN feature extraction
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.resnet(x)
        x = x.view(batch_size, seq_length, -1)

        # LSTM for temporal modeling
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size*2)

        # Attention mechanism
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(2)  # (batch_size, seq_len, 1)

        # Apply attention to LSTM output
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_size*2)

        # Classification with additional FC layer
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha)

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        return loss.mean()

def train_model(model, model_dir, criterion, optimizer, dataloaders, dataset_sizes, scheduler=None, device="cpu",
                num_epochs=20, grad_clip_val=1.0, patience=5, plot_path=None):
    """
    Enhanced training function with:
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling
    - Better tracking of metrics
    """
    model = model.to(device)
    since = time.time()

    # Initialize best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    no_improve_epoch = 0

    # Initialize metrics tracking
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []  # Track F1 score

    # Loop over epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only in training
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate epoch metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Calculate F1 score for validation
            if phase == 'val':
                val_report = classification_report(all_labels, all_preds, output_dict=True)
                epoch_f1 = val_report['macro avg']['f1-score']
                val_f1_history.append(epoch_f1)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, model_dir)
                    print(f"New best model saved with accuracy: {best_acc:.4f}")
                    no_improve_epoch = 0
                elif epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    no_improve_epoch = 0
                else:
                    no_improve_epoch += 1

            # Step scheduler if provided (after validation)
            if phase == 'val' and scheduler is not None:
                scheduler.step(epoch_loss)  # ReduceLROnPlateau

        # Early stopping
        if no_improve_epoch >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print()

    # Training complete
    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training curves
    if plot_path:
        plt.figure(figsize=(16, 6))

        # Convert accuracy histories to CPU numpy arrays
        train_acc_cpu = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_acc_history]
        val_acc_cpu = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in val_acc_history]

        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(train_loss_history, 'b-', label='Training Loss')
        plt.plot(val_loss_history, 'r-', label='Validation Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 3, 2)
        plt.plot(train_acc_cpu, 'b-', label='Training Accuracy')
        plt.plot(val_acc_cpu, 'r-', label='Validation Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot F1 score
        plt.subplot(1, 3, 3)
        plt.plot(val_f1_history, 'g-', label='Validation F1 Score')
        plt.title('F1 Score vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Training plots saved to {plot_path}")
        plt.close()

    return model

def test_model(model, dataloaders, device, class_names, plot_path=None, pr_curve_path=None):
    """
    Enhanced test function with:
    - Precision-Recall curve
    - More detailed metrics
    - Confusion matrix visualization
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # For PR curve

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate additional metrics
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Print overall metrics
    print(f"\nOverall Test Results:")
    print(f"Accuracy: {class_report['accuracy']:.4f}")
    print(f"Macro F1: {class_report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {class_report['weighted avg']['f1-score']:.4f}\n")

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
    plt.close()

    # Generate Precision-Recall curve (for binary classification)
    if len(class_names) == 2 and pr_curve_path:
        violence_probs = all_probs[:, 1]  # Assuming violence is class 1

        precision, recall, _ = precision_recall_curve(all_labels, violence_probs)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(pr_curve_path)
        print(f"Precision-Recall curve saved to {pr_curve_path}")
        plt.close()

    return all_preds, all_labels, cm