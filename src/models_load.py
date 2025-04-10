### src/models_laod.py


# IMPORTS
# __________________________________________________________________
import torch
from transformers import BertTokenizer, pipeline
from src.models_def import BertClassifier, ResNetModel, ResNetLSTMModel

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model_path = "./models/bert.pth"
    resnet_lstm_model_path = "./models/resnet50-lstm5.pt"  # Your ResNet-LSTM model
    class_names = ['Safe', 'Violence']  # Updated class names

    # Load BERT model (unchanged)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertClassifier().to(device)
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))
    bert_model.eval()

    # Load Whisper model (unchanged)
    whisper_model = pipeline("automatic-speech-recognition", "openai/whisper-tiny.en", torch_dtype=torch.float16, device=device)

    # Load ResNet-LSTM model
    # resnet_lstm_model = ResNetLSTMModel(num_classes=len(class_names), device=device)
    # # To this (only if you completely trust the model file source):
    # resnet_lstm_model.load_state_dict(torch.load(resnet_lstm_model_path, map_location=device, weights_only=False))
    resnet_lstm_model = torch.load(resnet_lstm_model_path, map_location=device, weights_only=False)
    resnet_lstm_model.eval()

    return {
        'tokenizer': tokenizer,
        'bert_model': bert_model,
        'whisper_model': whisper_model,
        'resnet_model': resnet_lstm_model,  # Now using ResNet-LSTM
        'class_names': class_names,
        'device': device
    }


### END
