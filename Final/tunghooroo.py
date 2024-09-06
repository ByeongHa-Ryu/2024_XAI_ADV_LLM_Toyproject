"""install things by requirements first """
from c_chatbot import * 


def run_agent_continuously(message, history=None):
    print("AI비서 < 텅후루 > 와의 대화를 시작합니다.")
    if history is None:
        history = []

    try:
        # chatbot 함수에 메시지와 이전 대화 기록을 전달하여 응답 생성
        response = chatbot(message=message, history=history)
        
        # 대화 기록에 현재 메시지와 응답 추가
        history.append({"message": message, "response": response})
        return response, history  # 응답과 업데이트된 대화 기록 반환
    
    except Exception as e:
        return f"에러가 발생했습니다: {e}", history
