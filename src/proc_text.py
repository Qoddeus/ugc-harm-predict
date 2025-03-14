### src/proc_text.py
### text analysis and highlighting


import numpy as np
import torch
import torch.nn.functional as F

def classify_text(transcription, bert_model, tokenizer, device):
    bert_model.eval()
    text = " ".join(segment["text"] for segment in transcription)
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits, attentions = bert_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    probs = F.softmax(logits, dim=-1)
    harmful_confidence = probs[0][1].item()
    safe_confidence = probs[0][0].item()
    label = "Harmful" if harmful_confidence > safe_confidence else "Safe"

    highlighted_text = highlight_toxic_words(text, inputs, attentions, tokenizer)

    return label, harmful_confidence, safe_confidence, highlighted_text

def highlight_toxic_words(text, inputs, attentions, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert token IDs to words
    attention_scores = attentions[-1].mean(dim=1).mean(dim=0).cpu().numpy()
    attention_scores = attention_scores.reshape(-1)[:len(tokens)]  # Ensure correct token length

    highlighted_text = []
    merged_tokens = []
    merged_attention = []

    for token, score in zip(tokens, attention_scores):
        if token.startswith("##"):
            merged_tokens[-1] += token[2:]  # Merge subwords
            merged_attention[-1] = np.maximum(merged_attention[-1], score)  # Keep max importance

        else:
            merged_tokens.append(token)
            merged_attention.append(score)

    # Determine dynamic thresholds
    threshold_high = np.percentile(merged_attention, 80)
    threshold_mid = np.percentile(merged_attention, 60)

    for token, score in zip(merged_tokens, merged_attention):
        if token in {".", ",", "!", "?", "'", '"', "'"}:
            highlighted_text.append(token)
            continue

        # Force highlight known toxic words
        if token.lower() in {"fuck", "bitch", "idiot", "stupid"}:
            highlighted_text.append(f"<span style='background-color:rgba(255, 0, 0, 1)'>{token}</span>")
            continue

        if np.any(score > threshold_high):
            score_scalar = np.max(score)  # Get the max score
            color = f"rgba(255, 0, 0, {max(score_scalar, 1):.2f})"  # High toxicity (red)
        elif score > threshold_mid:
            score_scalar = np.max(score)  # Get the max score
            color = f"rgba(255, 165, 0, {max(score_scalar, 1):.2f})"  # Mild toxicity (orange)
        else:
            color = "black"

        highlighted_text.append(f"<span style='color:{color}'>{token}</span>")

    return " ".join(highlighted_text)


### end
