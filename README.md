# Fake News Detection with RAG

Este projeto implementa um sistema de detecção de fake news utilizando **Retrieval-Augmented Generation (RAG)**. Ele combina um *retriever* de documentos, um *generator* com análise detalhada (*chain of thought*) e uma avaliação de desempenho.

## Requisitos

- Python 3.10.12
- Conda (para gerenciamento de ambientes)
- Chave de API para o **Groq** (para conectar ao cliente)

## Passo a Passo

### 1. Clonar o Repositório

Primeiro, clone o repositório para o seu ambiente local:

```bash
git clone https://github.com/jpamatos/fake-news-detector-rag.git
cd fake-news-detector-rag
```

### 2. Criar o Ambiente Conda

Crie um ambiente Conda chamado `fake_news`:

```bash
conda create --name fake_news python=3.10.12 --no-default-packages
conda activate fake_news
```

### 3. Instalar Dependências

Instale os pacotes necessários listados em `requirements.txt` usando `pip`:

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Certifique-se de definir a chave de API **Groq** como uma variável de ambiente. Adicione a chave no seu terminal:

```bash
export GROQ_KEY="sua_chave_de_api_groq"
```

### 5. Executar o Script

Execute o script principal `main.py` para carregar o dataset, criar os chunks, calcular os embeddings, e executar a avaliação:

```bash
python main.py
```

## Estrutura do Código

### Arquivos e Pastas Principais

- `main.py`: Script principal para executar o fluxo de detecção e avaliação.
- `rag/retriever.py`: Contém a classe `Retriever` para recuperação de documentos e gerenciamento do índice FAISS.
- `rag/generator.py`: Contém a classe `Generator`, que gera predições usando uma análise *chain of thought* com o cliente **Groq**.
- `evaluation/performance_evaluator.py`: Avalia a precisão, recall e F1-score das predições.
- `dataset/liar_dataset_processor.py`: Processa o dataset **LIAR** e realiza o chunking dos textos.

### Fluxo do Script `main.py`

1. **Processamento do Dataset**: Carrega o dataset LIAR e cria chunks de texto.
2. **Criação de Embeddings**: Usa o `Retriever` para calcular os embeddings e salva o índice FAISS.
3. **Carregamento do Índice**: Carrega o índice FAISS para recuperação de documentos.
4. **Conexão com o Cliente Groq**: Usa a chave de API para conectar ao cliente **Groq**.
5. **Geração de Predição**: Cria o `Generator` e faz predições de rótulo (verdadeiro ou falso) para uma notícia.
6. **Avaliação do Modelo**: Executa a avaliação usando um dataset de teste com precisão, recall e F1-score.

## Exemplo de Uso

O script `main.py` exibirá a previsão de veracidade para uma notícia de exemplo e calculará as métricas de desempenho para o dataset de teste.

### Saída Esperada

```plaintext
Predicted Label (1 = True, 0 = Fake): 1
Precisão: 0.89
Recall: 0.85
F1-Score: 0.87
```