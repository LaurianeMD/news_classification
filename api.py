# import torch
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from news_dataset import NewsDataset
# from utils import load_model, predict_category

# # Initialize FastAPI app
# app = FastAPI()

# # Load dataset and model
# dataset = NewsDataset(csv_file="./inshort_news_data.csv", max_length=100)
# num_classes = len(dataset.labels_dict)
# model_path = './models/trained_model.pth'  # Path to your trained model
# model = load_model(model_path, num_classes)
# labels_dict = dataset.labels_dict

# # Tokenizer initialization
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# # Define Pydantic model for input data
# class RequestPost(BaseModel):
#     headline: str
#     article: str

# @app.get("/")
# def read_root():
#    return {"Hello": "World"}

# # Define endpoint for prediction
# @app.post("/predict/")
# def prediction(request: RequestPost):
#     try:
#         category, score = predict_category(request.headline, request.article, model, labels_dict)
#         return {"category": category, "score": score}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from news_dataset import NewsDataset
from utils import load_model, predict_category

# Initialize FastAPI app
app = FastAPI()

# Load dataset and model
dataset = NewsDataset(csv_file="./inshort_news_data.csv", max_length=100)
num_classes = len(dataset.labels_dict)
model_path = './models/trained_model1.pth'  # Path to your trained model
model = load_model(model_path, num_classes)
labels_dict = dataset.labels_dict

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define Pydantic model for input data
class RequestPost(BaseModel):
    headline: str
    article: str

@app.get("/")
def read_root():
   return {"Hello": "World"}

# Define endpoint for prediction
@app.post("/predict/")
def prediction(request: RequestPost):
    try:
        category, score = predict_category(request.headline, request.article, model, labels_dict)
        return {"category": category, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
