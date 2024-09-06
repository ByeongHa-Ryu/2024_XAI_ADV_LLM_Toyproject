"""install things by requirements first """

from dotenv import load_dotenv
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from contextlib import redirect_stdout
import sys
import io
import os
import pickle
from PIL import Image 

from prompts import *
from a_login import *
import pickle 

load_dotenv()

# ## 이 부분 login + DB 로 연결 필요 
# df = pd.read_excel('data/sampling1.xlsx')

## LLM 
llm = ChatOpenAI(temperature=0.1)

# # pandas dataframe agent 
# agent = create_pandas_dataframe_agent(
#     llm=llm,                           
#     df= global_user_data,                             
#     verbose=False,                      
#     agent_type=AgentType.OPENAI_FUNCTIONS,
#     output_parser=StrOutputParser(),   
#     allow_dangerous_code=True 
# )
agent = None  # agent는 초기화 단계에서 정의할 필요 없음


def initialize_agent(user_data):
    # 로그인 후 전달된 user_data를 사용
    if user_data is None:
        raise ValueError("User data is not provided. Please log in first.")
    
    llm = ChatOpenAI(temperature=0.1)

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=user_data,  # 여기서 global_user_data 대신 user_data 사용
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        output_parser=StrOutputParser(),
        allow_dangerous_code=True)
    return agent


# 카드 추천 함수
def card_recommendation():
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

        prompt_template = """당신은 카드 추천 시스템입니다. 
        PDF 문서의 내용을 기반으로 사용자의 소비 패턴과 요구에 맞는 카드를 추천해 주세요.
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
    except Exception as e:
        return f"QA 체인 설정 중 오류가 발생했습니다: {e}"


# 챗봇 기능 통합 함수
def chatbot(message, history=None):
    try:
        # LLM을 사용하여 질문이 분석 관련인지 확인
        #print('1')
        classification_response = llm.invoke( input=analysis_check_prompt.format(input_query=message) )
        #print('2')
        classification_text = classification_response.content.strip()
        #print('3')


        # 1. 거래내역 분석 - LLM
        if "분석 관련" in classification_text:
            # print('cls 임시 print문 : 분석')
            # # 에이전트를 사용하여 질문에 대한 기본 분석 수행
            # agent_send = llm.invoke(
            #     input=agent_helping_prompt.format(user_query=message)
            # )
            
            # f = io.StringIO()
            # with redirect_stdout(f):
            #     analysis_result = agent.invoke(
            #         name="텅후루",
            #         role="understand the user query espicially on meaning that contained on word and do the right analysis.",
            #         input=agent_send)
                
            # verbose_output = f.getvalue()
            # # LLM을 사용하여 에이전트 결과에 대한 후처리 수행
            # final_response = llm.invoke(
            #     input=post_processing_prompt.format(
            #         verbose_output = verbose_output,
            #         analysis_result = analysis_result
            #         )
            # )
            # final_text = final_response.content.strip()
            
            analysis_result = agent.run(
                name="텅후루",
                role="사용자의 소비내역을 기반으로 여러가지 질문에 답하고 분석 정보를 제공하는 것",
                input=message)
            
            # LLM을 사용하여 에이전트 결과에 대한 후처리 수행
            final_response = llm.invoke(
                input=post_processing_prompt.format(analysis_result=analysis_result)
            )
            final_text = final_response.content.strip()
        
        

        # 2. 카드 추천 - 따로 함수에서 LLM
        elif "카드 관련" in classification_text:
            print('cls 임시 print문 : 카드관련')
            # 카드 추천 관련 질문은 qa_chain을 통해 처리
            final_text = card_recommendation("qa_chain", question=message)

        

        # 3. 일반 질문 - LLM -> main_prompt
        else:
            print('cls 임시 print문 : 일반 답변')
            # 분석과 관련 없는 일반 질문은 LLM이 직접 답변
            final_response = llm.invoke(
                input=main_prompt.format(input_query=message)
            )
            final_text = final_response.content.strip()


    except Exception as e:
        final_text = f"에러가 발생했습니다: {e}"
    history.append((message, final_text))
    return history  