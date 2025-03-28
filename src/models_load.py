### src/models_laod.py


# IMPORTS
# __________________________________________________________________
import torch
from transformers import BertTokenizer, pipeline
from src.models_def import BertClassifier, ResNetModel

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model_path = "./models/bert.pth"
    resnet_model_path = "./models/resnet50.pth"
    class_names = ['nsfw', 'safe', 'violence']

    # Load BERT model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertClassifier().to(device)
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))
    bert_model.eval()

    # Load Whisper model
    whisper_model = pipeline("automatic-speech-recognition", "openai/whisper-tiny.en", torch_dtype=torch.float16, device=device)

    # Load ResNet model
    resnet_model = ResNetModel(num_classes=len(class_names), device=device)
    state_dict = torch.load(resnet_model_path, map_location=device)
    resnet_model.load_state_dict(state_dict, strict=False)
    resnet_model.eval()

    return {
        'tokenizer': tokenizer,
        'bert_model': bert_model,
        'whisper_model': whisper_model,
        'resnet_model': resnet_model,
        'class_names': class_names,
        'device': device
    }


### END
