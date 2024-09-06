import gradio as gr
from a_login import *
from b_function_visualization import *
from c_chatbot import *

# Gradio 함수
def visualize_top_categories(k):
    plt = plot_top_category(df, k)
    return plt

def visualize_monthly_consumption(year):
    plt = plot_monthly_consumption(df, year)
    return plt

def recommend_card():
    response = fix_card_recommendation()
    return response


# ============================================================================================================

with gr.Blocks() as demo:
    gr.Markdown("# 통장이 텅텅 돈이 후루룩")

    with gr.Row():
        gr.Markdown("## 사용자 거래 데이터 및 로그인 시스템")
        username_input = gr.Textbox(label="사용자 이름", placeholder="사용자 이름을 입력하세요")
        account_number_input = gr.Textbox(label="계좌 번호", placeholder="계좌 번호를 입력하세요")
        login_button = gr.Button("로그인")
        output_text = gr.Textbox(label="결과", interactive=False)

# ============================================================================================================
    with gr.Row():  # 1행 3열로 구성
        with gr.Column():
        # 1행 1열: Tab 구성 => 지출별 top 카테고리 시각화 & 월별 소비내역 시각화
            with gr.Tabs() as tabs:
                with gr.Tab(label="지출별 Top 카테고리 시각화"):
                    top_category_plot = gr.Image(label="지출별 Top 카테고리 시각화", visible=False)
                with gr.Tab(label="월별 소비내역 시각화"):
                    # 연도를 선택하는 라디오 버튼 추가
                    year_radio = gr.Radio(choices=[2022, 2023, 2024], label="연도 선택", value=2023)
                    monthly_consumption_plot = gr.Plot(label="월별 소비내역 시각화", visible=True)

        # 1행 2열: 지출 top으로 도출되는 별명 출력
        with gr.Column():
            top_category_character = gr.Textbox(label=" 별명", visible=False, interactive=False)

        # 1행 3열: 카드 추천 함수로부터 도출되는 카드 추천 설명
        with gr.Column():
            card_recommendation_output = gr.Textbox(label="카드 추천 결과", visible=False, interactive=False)
        
        # 연도 선택 라디오 버튼이 변경되면 월별 소비내역 시각화 업데이트
    def update_monthly_consumption(year):
        return plot_monthly_consumption(df, year)

    year_radio.change(
        fn=update_monthly_consumption,
        inputs=year_radio,
        outputs=monthly_consumption_plot
    )


# ============================================================================================================
    chatbot_column = gr.Column(visible=False)
    with chatbot_column:
        # 하단에 챗봇 (초기에는 표시하지 않음)
        iface = gr.ChatInterface(
            chatbot,
            title="텅후루 톡",
            description="안녕하세요! 챗봇 가계부 텅후룩입니다!",
            theme="default",
            examples=[ ["고정지출을 분석해줘"], ["다음달 예상 지출 내역을 알려줘"], ['나의 지출 패턴을 분석해줘'], 
                      ["내 소비 패턴에 맞는 카드를 추천해줘"], ["포인트 적립이 높은 카드는 무엇인가요?"] ],
            retry_btn="다시보내기 ↩",
            undo_btn="이전챗 삭제 ❌",
            clear_btn="전챗 삭제 💫"
        )

# ============================================================================================================
    # login_button 클릭 시 login 함수 호출
    login_button.click(
        fn=login,
        inputs=[username_input, account_number_input],
        outputs=[output_text, 
                 top_category_plot, year_radio, 
                 monthly_consumption_plot, 
                 top_category_character,
                 chatbot_column ],
        queue=False
    )

# Gradio 앱 실행
demo.launch(debug=True, share=True)