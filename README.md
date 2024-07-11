---
title: News_article_classification_bert
app_file: main.py
sdk: gradio
sdk_version: 4.37.2
---
# news_classification
News Article Classification: Combining Headlines and Articles to Categorize News

# Classification des Catégories de News avec BERT

## Classification des Catégories de News avec BERT

Ce projet vise à classifier automatiquement les catégories de nouvelles à partir des titres et du contenu des articles en utilisant un modèle BERT préalablement entraîné.

## Contenu du Projet

- `bert_classification.py` : Contient la définition du modèle `CustomBert` utilisé pour la classification.
- `news_dataset.py` : Implémente la classe `NewsDataset` pour charger et prétraiter le dataset de nouvelles.
- `utils.py` : Fournit des fonctions utilitaires pour charger le modèle entraîné et effectuer des prédictions.
- `main.py` : Script principal pour l'entraînement du modèle et l'évaluation sur un dataset divisé.
- `api.py` : Implémente une API web à l'aide de FastAPI pour permettre la prédiction des catégories de nouvelles en temps réel.

## Installation des Dépendances

Assurez-vous d'avoir Python 3.7+ installé ainsi que les packages nécessaires :

pip install -r requirements.txt

## Entraînement du Modèle
Pour entraîner le modèle, exécutez main.py. Assurez-vous d'avoir un fichier CSV inshort_news_data.csv avec les colonnes news_headline et news_article.

python main.py


## Détails de l'Entraînement

Batch Size : 8 (par défaut)
Epochs : 3 (par défaut)
Précision : 96%, Perte : 0.1 après l'entraînement.
Modèle sauvegardé à ./models/trained_model1.pth.

## Utilisation de l'API Web
Pour utiliser l'API web pour la prédiction des catégories de news :

Lancez l'API avec FastAPI en exécutant api.py:

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

Ce projet démontre l'utilisation de modèles NLP avancés comme BERT pour la classification automatique des données textuelles, spécifiquement dans le domaine des nouvelles et articles. Pour plus d'informations, n'hésitez pas à consulter la documentation des librairies utilisées comme Transformers et FastAPI.






