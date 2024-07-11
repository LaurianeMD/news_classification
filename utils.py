import torch
from transformers import AutoTokenizer
import torch.nn as nn
from bert_classification import CustomBert  # Importer le modèle depuis le fichier bert_classification.py

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def load_model(model_path, num_classes):
    model = CustomBert(n_classes=num_classes)  # Adapter ici le nombre de classes
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_category(headline, article, model, labels_dict, max_length=100):
    text = headline + " " + article
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probabilities = nn.Softmax(dim=1)(output)
        _, pred = torch.max(probabilities, dim=1)
        score = probabilities[0][pred].item()

        inv_labels_dict = {v: k for k, v in labels_dict.items()}
        category = inv_labels_dict[pred.item()]

    score = round(score, 2) 

    return category, score
