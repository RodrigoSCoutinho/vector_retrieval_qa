from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline  
from transformers import pipeline  

def main():
    file_paths = ["data/texto_ia.txt", "data/texto_redes_neurais.txt"]
    
    documents = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
            print(f"Carregado arquivo: {path}")

    docs = [Document(page_content=doc) for doc in documents]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    print(f"Documento dividido em {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embeddings criados.")

    db = FAISS.from_documents(texts, embeddings)
    print("Índice FAISS criado.")

    db.save_local("faiss_index")
    print("Índice salvo em 'faiss_index'.")

    retriever = db.as_retriever()

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128,
        temperature=0
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\nSistema de Perguntas e Respostas ativado!")
    print("Digite 'sair' para encerrar.\n")

    while True:
        query = input("Faça sua pergunta: ")
        if query.lower() == "sair":
            print("Encerrando o sistema.")
            break
        
        try:
            resposta = qa.run(query)  
            print("Resposta:", resposta)
        except Exception as e:
            print("Erro ao processar a pergunta:", e)
        
        print("-" * 40)

if __name__ == "__main__":
    main()
