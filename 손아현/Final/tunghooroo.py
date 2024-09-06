"""install things by requirements first """
from chatbot import * 

from chatbot import * 

def run_agent_continuously():
    print("AI비서 < 텅후루 > 와의 대화를 시작합니다.")

    while True:
        input_query = input("질문: ")
        
        if input_query.lower() == "exit":
            print("대화를 종료합니다.")
            break
        
        try:
            response = chatbot(message=input_query)  # chatbot 함수의 반환값 저장
            print("AI:", response)  # 반환값을 출력
        except Exception as e:
            print("에러가 발생했습니다:", e)

if __name__ == '__main__':
    run_agent_continuously()
