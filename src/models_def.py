# src/models_def.py


# IMPORTS
# __________________________________________________________________
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertForSequenceClassification

# Define the new attention-based classifier
class BertClassifier(nn.Module):
  def __init__(self, dropout_rate=0.3):
    super(BertClassifier, self).__init__()
    self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=True)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    logits = self.dropout(outputs.logits)
    attentions = outputs.attentions  # Extract attention scores
    return logits, attentions

class ResNetModel(nn.Module):
  def __init__(self, num_classes=3, device="cuda" if torch.cuda.is_available() else "cpu"):
    super(ResNetModel, self).__init__()
    self.model = models.resnet50(pretrained=True)
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer
    self.model.to(device)

  def forward(self, x):
    return self.model(x)


# Add this new class to your models_def.py
class ResNetLSTMModel(nn.Module):
  def __init__(self, num_classes=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    super(ResNetLSTMModel, self).__init__()
    # Load pre-trained ResNet50
    resnet = models.resnet50(pretrained=True)
    # Remove the final fully connected layer
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.lstm = nn.LSTM(
      input_size=resnet.fc.in_features,  # 2048 features from ResNet
      hidden_size=512,
      num_layers=2,
      batch_first=True,
      bidirectional=True
    )
    self.fc = nn.Linear(512 * 2, num_classes)  # *2 for bidirectional
    self.device = device
    self.to(device)

  def forward(self, x):
    # x shape: (batch_size, sequence_length, C, H, W)
    batch_size, seq_length, C, H, W = x.size()

    # Reshape to (batch_size * seq_length, C, H, W)
    x = x.view(batch_size * seq_length, C, H, W)

    # Get features from ResNet
    x = self.resnet(x)
    x = x.view(batch_size, seq_length, -1)  # (batch_size, seq_length, 2048)

    # LSTM processing
    lstm_out, _ = self.lstm(x)

    # Take the output from the final timestep
    out = self.fc(lstm_out[:, -1, :])
    return out

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


# END
# __________________________________________________________________
