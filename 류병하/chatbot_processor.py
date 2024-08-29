import os
from dotenv import load_dotenv
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

df = pd.read_csv('전처리_내역.csv')

# 가상환경 확인 및 env 파일 Load 
virtual_env = os.environ.get('VIRTUAL_ENV')
if virtual_env:
    print("Virtual environment is active.")
    print("Virtual Environment Path:", virtual_env)
else:
    print("No virtual environment is active.")

print('.env loaded : ',load_dotenv())

# LLM 초기화
llm = ChatOpenAI(temperature=0.1)  

# 사용자 정의 프롬프트 구성 (LLM 후처리용)

post_processing_prompt = PromptTemplate(
    input_variables=["analysis_result"],
    template="""
    다음은 은행 고객의 AI 비서가 생성한 분석 결과입니다:
    {analysis_result}
    이 결과를 바탕으로 고객이 쉽게 이해할 수 있도록 요약하고, 추가적으로 유용할 만한 정보를 제공하세요.
    """
)

# 에이전트 생성 (데이터프레임 분석 수행)

agent = create_pandas_dataframe_agent(
    llm=llm,                           
    df=df,                             
    verbose=True,                      
    agent_type=AgentType.OPENAI_FUNCTIONS,
    output_parser=StrOutputParser(),   
    allow_dangerous_code=True         
)

main_prompt = ' 당신은 이름은 텅후루 입니다. 당신의 역할은 은행 고객의 AI비서가 생성한 분석 결과를 활용하여 대답하는 것입니다.'

# 질문이 분석과 관련된지 여부를 확인하는 LLM 프롬프트

analysis_check_prompt = PromptTemplate(
    input_variables=["input_query"],
    template="""
    아래의 질문이 데이터 분석과 관련이 있는지 아닌지를 판단해 주세요. 
    질문이 분석 관련이면 '분석 관련'이라고 대답하고, 그렇지 않으면 '일반 질문'이라고 대답하세요.

    질문: {input_query}
    답변: 
    """
)

def run_agent_continuously():
    print("AI비서 < 텅후루 > 와의 대화를 시작합니다.")

    while True:
        input_query = input("질문: ")
        
        if input_query.lower() == "exit":
            print("대화를 종료합니다.")
            break
        
        try:
            # LLM을 사용하여 질문이 분석 관련인지 확인
            classification_response = llm.invoke(
                input=analysis_check_prompt.format(input_query=input_query)
            )

            # classification_response는 AIMessage 객체로 반환됩니다. content 속성을 사용하여 텍스트를 추출
            classification_text = classification_response.content.strip()
            
            if "분석 관련" in classification_text:
                # 에이전트를 사용하여 질문에 대한 기본 분석 수행
                analysis_result = agent.run(
                    name="텅후루",
                    role="사용자의 소비내역을 기반으로 여러가지 질문에 답하고 분석 정보를 제공하는 것",
                    input=input_query
                )
                
                # LLM을 사용하여 에이전트 결과에 대한 후처리 수행
                final_response = llm.invoke(
                    input=post_processing_prompt.format(analysis_result=analysis_result)
                )
                final_text = final_response.content.strip()
            else:
                # 분석과 관련 없는 일반 질문은 LLM이 직접 답변
                final_response = llm.invoke(
                    input=main_prompt.format(input_query=input_query)
                )
                
                final_text = final_response.content.strip()
            
            print("에이전트 응답:", final_text)
        
        except Exception as e:
            print("에러가 발생했습니다:", e)

# 대화 루프 실행
run_agent_continuously()
