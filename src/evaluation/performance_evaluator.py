import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluator:
    def __init__(self, generator, k: int=5) -> None:
        """
        Inicializa o avaliador com um objeto Generator e o número de resultados (k).
        
        Args:
            generator (Generator): Objeto Generator que realiza as predições.
            k (int): Número de documentos similares a serem recuperados.
        """
        self.generator = generator
        self.k = k

    def evaluate_performance(self, test_data: pd.DataFrame) -> dict:
        """
        Calcula precisão, recall e F1-score com base no test_data fornecido.
        
        Args:
            test_data (pd.DataFrame): DataFrame com colunas 'title', 'text' e 'label'.
        
        Returns:
            dict: Dicionário com as métricas de precisão, recall e F1-score.
        """
        true_labels = []
        predicted_labels = []

        for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
            title = row['title']
            text = row['text']
            true_label = row['label']  # Rótulo verdadeiro: 1 para verdadeiro, 0 para fake news

            # Obter a previsão usando o Generator
            predicted_label = self.generator.predict_label(title, text, k=self.k)

            # Armazenar os rótulos verdadeiro e previsto
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

        # Calcular as métricas de avaliação
        precision = precision_score(true_labels, predicted_labels, average='binary')
        recall = recall_score(true_labels, predicted_labels, average='binary')
        f1 = f1_score(true_labels, predicted_labels, average='binary')

        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
