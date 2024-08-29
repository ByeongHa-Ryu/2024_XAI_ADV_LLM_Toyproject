import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pickle
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_documents(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

def create_or_update_vectordb(pdf_path, db_path="vectordb.pkl"):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            vectordb = pickle.load(f)
    else:
        vectordb = FAISS.from_texts([], embeddings)

    text = load_pdf(pdf_path)
    chunks = split_documents(text)
    vectordb.add_texts(chunks)

    with open(db_path, "wb") as f:
        pickle.dump(vectordb, f)

    return vectordb

def get_qa_chain():
    with open("vectordb.pkl", "rb") as f:
        vectordb = pickle.load(f)

    # 여기를 수정합니다
    texts = [doc.page_content for doc in vectordb.docstore._dict.values()]
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 5

    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    prompt_template = """당신은 PDF 문서의 내용을 기반으로 질문에 답변하는 AI 어시스턴트입니다. 
    주어진 컨텍스트를 사용하여 질문에 정확하고 간결하게 답변해 주세요.

    컨텍스트:
    {context}

    질문: {question}

    답변:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def ask_question(question):
    qa_chain = get_qa_chain()
    result = qa_chain({"query": question})
    return result['result']