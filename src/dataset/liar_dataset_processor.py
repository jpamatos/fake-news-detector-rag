from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LiarDatasetProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50) -> None:
        # Parâmetros de chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        # Carregar o dataset LIAR
        self.dataset = load_dataset("liar")

    def chunk_data(self, split='train'):
        """
        Realiza o chunking dos dados do split especificado (train, test ou validation).
        
        Args:
            split (str): O split do dataset para processar (train, test ou validation).
            
        Returns:
            List[Dict]: Uma lista de chunks, cada um com texto e metadados.
        """
        chunks = []
        for entry in self.dataset[split]:
            statement = entry['statement']
            context = f"{statement} - {entry['subject']} - {entry['speaker']}"

            # Dividir o contexto em chunks
            document_chunks = self.text_splitter.split_text(context)
            for chunk in document_chunks:
                chunks.append({"text": chunk, "metadata": entry})
        return chunks