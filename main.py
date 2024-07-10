# main.py

import gradio as gr
from utils import load_model, predict_category
from news_dataset import NewsDataset  # Importez NewsDataset depuis news_dataset.py

def launch_app():
    dataset = NewsDataset(csv_file="./inshort_news_data.csv", max_length=100)
    num_classes = len(dataset.labels_dict)
    model_path = './models/trained_model.pth'  # Chemin vers le modèle entraîné
    model = load_model(model_path, num_classes)  # Charger le modèle entraîné avec le bon nombre de classes

    labels_dict = dataset.labels_dict

    def predict_function(headline, article):
        return predict_category(headline, article, model, labels_dict)

    iface = gr.Interface(
        fn=predict_function,
        inputs=["text", "text"],
        outputs="text",
        title="News Category Classification",
        description="Enter a headline and an article to classify its category."
    )

    iface.launch()

if __name__ == "__main__":
    launch_app()
