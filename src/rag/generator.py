class Generator:
    def __init__(self, model: str="llama3-8b-8192", retriever=None, client=None) -> None:
        """
        Inicializa o gerador com um modelo de LLM, um objeto de retriever e o cliente de API.
        
        Args:
            model (str): Nome do modelo LLM para geração.
            retriever (Retriever): Objeto retriever para buscar documentos relacionados.
            client (obj): Cliente de API para gerar respostas.
        """
        self.model = model
        self.retriever = retriever
        self.client = client

    def interpret_results_with_chain_of_thought(self, similar_docs: list) -> int:
        """
        Interpreta os documentos recuperados usando uma lógica de chain of thought
        para decidir se a notícia é verdadeira ou falsa.
        
        Args:
            similar_docs (List): Lista de documentos recuperados pelo retriever.
        
        Returns:
            int: 1 para verdadeiro, 0 para falso.
        """
        system_prompt = (
            "You are an assistant that verifies if a news is fake or true by analyzing retrieved evidence step-by-step. "
            "Consider each document separately. "
            "For each document, analyze the content and label it as supporting truth or indicating falsehood. "
            "Finally, based on the analysis of all documents, determine if the news is likely true or fake."
        )

        user_prompt = "Here are the documents retrieved for this news:\n\n"
        for i, doc in enumerate(similar_docs):
            content = doc.page_content[:200]  # Analisando os primeiros 200 caracteres para resumo
            label = "true" if (int(doc.metadata['label']) == 5 or int(doc.metadata['label']) == 4) else "fake"
            user_prompt += f"Document {i+1} (Label: {label}): {content}\n\n"

        user_prompt += (
            "Analyze each document and decide if it supports the truthfulness of the news or suggests it might be fake. "
            "After reviewing each document, provide a final conclusion on whether the news is likely true or fake."
        )

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model
        )

        result_text = response.choices[0].message.content.strip().lower()
        return 1 if "true" in result_text else 0

    def predict_label(self, title: str, text: str, k: int=5) -> int:
        """
        Gera uma previsão de rótulo (1 para verdadeiro, 0 para falso) para a notícia fornecida.
        
        Args:
            title (str): Título da notícia.
            text (str): Texto do conteúdo da notícia.
            k (int): Número de documentos similares a serem recuperados.
        
        Returns:
            int: 1 para verdadeiro, 0 para falso.
        """
        # Recupera os documentos mais similares usando o retriever
        query = f"Title: {title}\n\nText: {text}"
        similar_docs = self.retriever.search(query, k=k)
        
        # Interpreta os resultados e faz uma previsão
        return self.interpret_results_with_chain_of_thought(similar_docs)
