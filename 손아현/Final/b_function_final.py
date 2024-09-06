from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from typing import List
import gradio as gr
from transformers import pipeline, BertTokenizer
from kobert_transformers import get_kobert_model
import requests
import sqlite3

import os
import fitz  # PyMuPDF for PDF processing
import pickle
from datetime import datetime, timedelta  # datetime 모듈 임포트

# Langchain 관련 라이브러리
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from dateutil.relativedelta import relativedelta

# 환경 변수 로드를 위한 라이브러리
from dotenv import load_dotenv

# Langchain 추가 라이브러리
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType



# .env 파일 로드
print('.env loaded : ', load_dotenv())

# 환경 변수에서 API 키 가져오기
API_KEY = os.getenv("OPENAI_API_KEY")

# API_KEY가 제대로 로드되었는지 확인
if not API_KEY:
    raise ValueError("API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정하세요.")

# LLM 초기화
llm = ChatOpenAI(temperature=0.1, openai_api_key=API_KEY)




# 1. 지출 top 카테고리 plot  &  2. 별명 추출
## 1-1. 지출 top 카테고리
def top_k_category(df,k):
    # recent n month time stamp 
    current_time = datetime.now()
    n = 5 
    timepoints = current_time - relativedelta(months=n)
    filtered_df = df[(df['거래일'] >= timepoints) & (df['거래일'] <= current_time)]
    # top k categories 
    top_k_category = filtered_df.groupby('mid')['출금금액'].agg('sum').sort_values(ascending=False).head(k).reset_index()
    # corresponds character 
    top_1_category = top_k_category.iloc[1].mid
    
    if top_1_category == '교통' : 
        Character = '대중교통맨'
        
    elif top_1_category == '음식' :
        Character = '푸드파이터'
        
    elif top_1_category == '패션' : 
        Character = '패셔니스타'
        
    else :
        Character = '첨 보는 스타일...'
        
    return top_k_category , Character


# 3. 월 별 소비내역 그래프 - 년도 input 받기 (라디오 버튼)
# 3-1. 월 별 소비내역 계산 함수
def montly_consumption(df, year):
    filtered_df = df[df['연도'] == year]
    montly_consumption = filtered_df.groupby('월')['출금금액'].agg('sum').reset_index()
    return montly_consumption
##근데 사실상 

# 4. 카드 추천 함수 (기존 유지, 이름만 변경 - (구)get_qa_chain )
def fix_card_recommendation(question="내 소비 패턴에 맞는 카드를 추천해줘"):
    try:
        with open("vectordb.pkl", "rb") as f:
            vectordb = pickle.load(f)

        texts = [doc.page_content for doc in vectordb.docstore._dict.values()]
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 5

        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

        prompt_template = """당신은 카드 추천 시스템입니다. PDF 문서의 내용을 기반으로 사용자의 소비 패턴과 요구에 맞는 카드를 추천해 주세요.
        주어진 컨텍스트를 사용하여 질문에 정확하고 간결하게 답변해 주세요.

        컨텍스트:
        {context}

        질문: {question}

        답변: ### 카드 추천:
        1. **추천 카드 이름**: {카드 이름}

        2. **추천 이유**: 
        - 사용자 소비 패턴 분석 결과, 이 카드는 {특정 지출 카테고리}에서 높은 포인트 적립률을 제공합니다.
        - 사용자가 자주 사용하는 {특정 상점 또는 서비스}에서 추가 할인 혜택을 받을 수 있습니다.
        - {사용자의 요구 사항 또는 선호도}에 따라, 이 카드는 {특정 혜택 예: 연회비 면제, 해외 결제 수수료 없음 등}이 있어 적합합니다.

        3. **카드 혜택 정보**:
        - **적립 혜택**: {예: 모든 구매 금액의 2% 적립, 특정 카테고리 5% 캐시백}
        - **할인 혜택**: {예: 특정 상점에서 10% 할인, 주유소 5% 할인}
        - **부가 혜택**: {예: 여행 보험, 공항 라운지 이용권, 무료 커피 쿠폰 등}
        - **연회비**: {예: 연 30,000원, 첫 해 연회비 무료 등}

        4. **추가 정보**:
        - **적용 조건**: {예: 월 최소 사용 금액 30만 원, 특정 기간 동안 한정}
        - **주의사항**: {예: 일부 혜택은 국내에서만 적용, 매월 적립 한도 제한 있음 등}

        ### 최종 추천:
        이 카드는 사용자의 소비 습관과 요구 사항을 가장 잘 충족하며, 다양한 혜택을 통해 경제적인 이점을 제공할 수 있습니다.
                
        """

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
        # 질문을 사용하여 QA 체인을 통해 카드 추천 결과 생성
        response = qa_chain.run(question)

        return response
    
    except Exception as e:
        return f"QA 체인 설정 중 오류가 발생했습니다: {e}"
    

if __name__ == "__main__":
    top_k_category()