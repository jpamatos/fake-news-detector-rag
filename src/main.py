import os

import pandas as pd
from groq import Groq

from rag.retriever import Retriever
from rag.generator import Generator
from evaluation.performance_evaluator import Evaluator
from dataset.liar_dataset_processor import LiarDatasetProcessor


def main() -> None:
    # Carrega o dataset e Cria os chunks
    liar_processor = LiarDatasetProcessor()
    train_chunks = liar_processor.chunk_data(split='train')

    # Calcula os embeddings e salva o faiss index
    retriever = Retriever(model_name="sentence-transformers/all-mpnet-base-v2")
    retriever.create_faiss_index(train_chunks)
    retriever.save_faiss_index("../data/liar_rag")

    # Carrega o faiss index
    retriever.load_faiss_index("../data/liar_rag")

    # Conecta com o cliente do Groq usando a API KEY
    client = Groq(api_key=os.getenv("GROQ_KEY"))

    # Cria o generator
    generator = Generator(model="llama3-8b-8192", retriever=retriever, client=client)

    # Teste com título, texto e o uso do retriever
    title = "The president increased taxes significantly in the past year."
    text = "The president's recent economic policies led to a notable increase in taxes as reported by officials."
    predicted_label = generator.predict_label(title, text, k=5)

    # Exibir o rótulo previsto
    print("Predicted Label (1 = True, 0 = Fake):", predicted_label)

    test_data = pd.read_csv("../data/WELFake_Dataset.csv")

    # Executar a avaliação usando test_data (DataFrame do pandas com 'title', 'text' e 'label')
    evaluator = Evaluator(generator=generator, k=5)
    metrics = evaluator.evaluate_performance(test_data)


if __name__ == "__main__":
    main()
