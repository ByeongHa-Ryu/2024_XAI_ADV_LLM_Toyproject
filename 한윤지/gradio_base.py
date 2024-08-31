import gradio as gr
import random
import openai
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
API_KEY = os.environ.get('OPENAI_API_KEY')

def get_transactions():
    return [
        {"date": "2024-08-01", "amount": 1000, "description": "급여"},
        {"date": "2024-08-02", "amount": -50, "description": "식료품"},
        {"date": "2024-08-03", "amount": -30, "description": "교통비"},
    ]

def get_card_products():
    return ['여행 특화 카드', '해외 사용 시 5% 할인, 항공 마일리지 적립']

def get_ai_response(prompt):
    response = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)
    return response.generate([prompt]).generations[0][0].text

# 거래 내역 분석 및 카드 추천
def analyze_transactions_and_recommend_card(transactions, card_products):
    transaction = get_transactions()
    card = get_card_products()
    prompt = f"""다음은 사용자의 계좌 거래 내역입니다:

{transaction}

다음은 사용 가능한 카드 상품 목록입니다:

{card}

이 거래 내역을 분석하고, 사용자의 소비 패턴에 가장 적합한 카드를 추천해주세요. 
분석 결과와 카드 추천 이유를 상세히 설명해주세요."""

    return get_ai_response(prompt)

# 상태를 저장할 변수들
user_info = {"name": "", "account_number": ""}
transactions = []
card_products = []

def chatbot(message, history):
    if not user_info["name"] or not user_info["account_number"]:
        if "," in message :
            name, account_number = message.split(", ")
            user_info["name"] = name
            user_info["account_number"] = account_number
            transactions.extend(get_transactions(user_info["account_number"]))
            card_products.extend(get_card_products())
            
            analysis_and_recommendation = analyze_transactions_and_recommend_card(transactions, card_products)
            return f"{user_info['name']}님, 계좌번호 {user_info['account_number']}의 분석 결과입니다.\n\n{analysis_and_recommendation}\n\n추가로 궁금한 점이 있으시면 자유롭게 질문해 주세요."
        else:
            return "이름과 계좌번호를 입력해주세요. (예: 이름: 홍길동, 계좌번호: 1234-5678)"
    
    # 자유 질문에 대한 응답
    prompt = f"""당신은 친절한 금융 가계부 챗봇입니다. 사용자의 질문에 친절하게 답하세요.
    사용자 질문: {message}."""

    return get_ai_response(prompt)



# Gradio 인터페이스 생성
iface = gr.ChatInterface(
    chatbot,
    textbox=gr.Textbox(placeholder="어떤 것을 분석해드릴까요?", container=False, scale=7),
    title="통장이 텅텅 돈이 후루룩",
    description="안녕하세요! 챗봇 가계부 텅후룩입니다! 이름과 계좌번호를 '이름,계좌번호' 형식으로 입력해주세요.",
    theme="default",
    examples=[["고정지출을 분석해줘"], ["다음달 예상 지출 내역을 알려줘"], ['나의 지출 패턴을 분석해줘']],
    retry_btn="다시보내기 ↩",
    undo_btn="이전챗 삭제 ❌",
    clear_btn="전챗 삭제 💫"
)

# 실행
iface.launch()