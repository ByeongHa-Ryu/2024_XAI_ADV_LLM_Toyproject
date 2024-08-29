import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_pdfs_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            text = load_pdf(file_path)
            documents.append({"content": text, "source": filename})
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    splits = []
    for doc in documents:
        docs = text_splitter.create_documents([doc["content"]], [{"source": doc["source"]}])
        splits.extend(docs)
    return splits

def create_or_update_vectordb(pdf_directory, db_path="vectordb.pkl"):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 기존 DB가 있는지 확인
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            vectordb = pickle.load(f)
        print("기존 벡터 DB를 로드했습니다.")
    else:
        vectordb = FAISS.from_documents([], embeddings)
        print("새로운 벡터 DB를 생성했습니다.")

    # 새로운 문서 로드 및 처리
    loaded_documents = load_pdfs_from_directory(pdf_directory)
    split_docs = split_documents(loaded_documents)

    # 새로운 문서를 DB에 추가
    vectordb.add_documents(split_docs)

    # 업데이트된 DB 저장
    with open(db_path, "wb") as f:
        pickle.dump(vectordb, f)

    print(f"벡터 DB가 업데이트되어 '{db_path}' 파일로 저장되었습니다.")

def main():
    pdf_directory = input("PDF 파일이 있는 디렉토리 경로를 입력하세요: ")
    create_or_update_vectordb(pdf_directory)

if __name__ == "__main__":
    main()