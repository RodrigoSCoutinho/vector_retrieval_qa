# Sistema de Recuperação de Informações com VectorDB e RetrievalQA

## Descrição

Este projeto implementa um sistema de recuperação de informações usando embeddings vetoriais e banco de dados FAISS, integrando a biblioteca LangChain e o módulo RetrievalQA. Ele permite responder perguntas com base em documentos de texto.

## Como usar

### Pré-requisitos

- Python 3.8 ou superior
- Acesso à internet (para baixar modelos e usar HuggingFaceHub)

### Instalação

1. Clone o repositório:

```bash
git clone https://github.com/RodrigoSCoutinho/vector_retrieval_qa.git
cd vector_retrieval_qa
```
2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

### Instale as dependências
```bash
pip install -r requirements.txt
```

### Executando o código
```
python main.py
```