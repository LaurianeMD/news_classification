---
title: News_article_classification_bert
app_file: main.py
sdk: gradio
sdk_version: 4.37.2
---
# news_classification
News Article Classification: Combining Headlines and Articles to Categorize News

# **News Classification Using BERT**  ||  [Link of the application on hugging face](https://huggingface.co/spaces/LaurianeMD/News_article_classification_bert)

This project utilizes BERT (Bidirectional Encoder Representations from Transformers) for classifying news articles into predefined categories. The model achieves an accuracy of 96% and a loss of 0.1 on the test dataset.

## **Dataset**
The dataset used in this project is inshort_news_data.csv, containing short news articles categorized into various topics.

## **Model Architecture**
The model architecture is based on a custom BERT model fine-tuned for sequence classification:

BERT Model: bert-base-uncased
Batch Size: 8
Optimizer: Adam with learning rate 2e-5
Loss Function: CrossEntropyLoss
Training
The model is trained for 3 epochs with the following steps:

**Data Preparation:** The dataset is tokenized using the BERT tokenizer and prepared as PyTorch DataLoader objects.

**Training:** The model is trained using stochastic gradient descent with backpropagation. During training, the loss is minimized and weights are updated iteratively.

**Evaluation:** After each epoch, the model is evaluated on a held-out validation set to measure accuracy and loss.

**Results**
Accuracy: 96%
Loss: 0.1
Usage
To use the trained model for inference:

Ensure all dependencies are installed (transformers, torch, fastapi, pydantic, etc.).
Load the model using torch.load() and the appropriate tokenizer.
Send POST requests to /predict/ endpoint with JSON containing headline and article fields to classify news articles.
How to Run
To run the FastAPI application:
uvicorn api:app --host localhost --port 8080

Navigate to http://localhost:8080/docs to interact with the API using Swagger UI.

---------------------------------------------------------------------------------------------------
***french***
# Classification des Catégories des articles avec BERT

Ce projet vise à classifier automatiquement les catégories de nouvelles à partir des titres et du contenu des articles en utilisant un modèle BERT préalablement entraîné.

## Contenu du Projet

- `bert_classification.py` : Contient la définition du modèle `CustomBert` utilisé pour la classification.
- `news_dataset.py` : Implémente la classe `NewsDataset` pour charger et prétraiter le dataset de nouvelles.
- `utils.py` : Fournit des fonctions utilitaires pour charger le modèle entraîné et effectuer des prédictions.
- `main.py` :  charge un modèle pré-entraîné pour la classification des catégories de nouvelles, crée une interface utilisateur web avec Gradio
  pour permettre aux utilisateurs de  soumettre des titres et des articles, et affiche la catégorie prédite pour ces nouvelles.
- `api.py` : Implémente une API web à l'aide de FastAPI pour permettre la prédiction des catégories de nouvelles en temps réel.

## Installation des Dépendances

Assurez-vous d'avoir Python 3.7+ installé ainsi que les packages nécessaires :

pip install -r requirements.txt

## Entraînement du Modèle
Pour entraîner le modèle, exécutez main.py. Assurez-vous d'avoir un fichier CSV inshort_news_data.csv avec les colonnes news_headline et news_article.

python main.py


## Détails de l'Entraînement <br>

Batch Size : 8 (par défaut) <br>
Epochs : 3 (par défaut) <br>
Précision : 96%, Perte : 0.1 après l'entraînement.<br>
Modèle sauvegardé à ./models/trained_model1.pth.<br>

## Utilisation de l'API Web
Pour utiliser l'API web pour la prédiction des catégories de news :<br>

Lancez l'API avec FastAPI en exécutant api.py:<br>

uvicorn api:app --host localhost --port 8080

Accédez à http://localhost:8080 dans votre navigateur pour vérifier que l'API est en ligne.
Envoyez des requêtes POST à [http://localhost:8080/predict/](http://localhost:8080/docs#/default/prediction_predict__post) avec les données d'entrée requises pour obtenir des prédictions de catégories de news.
Exemple de requête JSON pour la prédiction :

json

{
  "headline": "50-year-old problem of biology solved by Artificial Intelligence",
  "article": "DeepMind's AI system 'AlphaFold' has been recognised as a solution to \"protein folding\", a grand challenge in biology for over 50 years. DeepMind showed it can predict how proteins fold into 3D shapes, a complex process that is fundamental to understanding the biological machinery of life. AlphaFold can predict the shape of proteins within the width of an atom."
}
Exemple de réponse attendue :

json

{
  "category": "Science",
  "score": 94.23
}
Assurez-vous d'avoir une connexion Internet active lors de l'exécution de l'API pour permettre le chargement du tokenizer BERT.








