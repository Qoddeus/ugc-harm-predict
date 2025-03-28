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


# END
# __________________________________________________________________
