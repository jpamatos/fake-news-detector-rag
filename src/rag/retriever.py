import torch
import numpy as np
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class Retriever:
    def __init__(self, model_name: str) -> None:
        """
        Inicializa o modelo de embeddings.
        
        Args:
            model_name (str): Nome do modelo de embeddings da Hugging Face.
        """
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.faiss_store = None

    def create_embeddings(self, chunks: list[dict]) -> list[np.ndarray]:
        """
        Cria embeddings para uma lista de chunks.
        
        Args:
            chunks (List[Dict]): Lista de chunks com texto e metadados.
            
        Returns:
            List[np.ndarray]: Lista de embeddings.
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings_list = []
        for text in tqdm(texts, desc="Gerando embeddings"):
            embedding = self.embedding_model.embed_query(text)
            embeddings_list.append(embedding)
        return texts, embeddings_list

    def create_faiss_index(self, chunks: list[dict]) -> None:
        """
        Cria e armazena um índice FAISS a partir dos chunks.
        
        Args:
            chunks (List[Dict]): Lista de chunks com texto e metadados.
        """
        texts, embeddings_list = self.create_embeddings(chunks)
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Cria o índice FAISS com os textos e embeddings
        self.faiss_store = FAISS.from_embeddings(
            zip(texts, embeddings_list),
            self.embedding_model,
            metadatas=metadatas
        )
        print("Índice FAISS criado com sucesso.")

    def save_faiss_index(self, directory: str="../../data/liar_rag") -> None:
        """
        Salva o índice FAISS localmente.
        
        Args:
            directory (str): Caminho para salvar o índice.
        """
        if self.faiss_store is not None:
            self.faiss_store.save_local(directory)
            print(f"Índice FAISS salvo em {directory}")
        else:
            print("Erro: O índice FAISS não foi criado.")
    
    def load_faiss_index(self, directory="liar_rag"):
        """
        Carrega um índice FAISS salvo localmente.
        
        Args:
            directory (str): Caminho para carregar o índice.
        """
        try:
            self.faiss_store = FAISS.load_local(directory, self.embedding_model, allow_dangerous_deserialization=True)
            print(f"Índice FAISS carregado de {directory}")
        except Exception as e:
            print(f"Erro ao carregar o índice FAISS: {e}")
    
    def search(self, query: str, k: int=5) -> list[dict]:
        """
        Busca no índice FAISS com base na consulta e retorna os chunks mais similares.
        
        Args:
            query (str): A consulta de texto.
            k (int): Número de resultados a retornar.
        
        Returns:
            List[Dict]: Lista dos k chunks mais similares e seus metadados.
        """
        if self.faiss_store is None:
            print("Erro: O índice FAISS não está carregado.")
            return []
        
        query_embedding = self.embedding_model.embed_query(query)
        similar_chunks = self.faiss_store.similarity_search_by_vector(query_embedding, k=k)
        
        return similar_chunks
