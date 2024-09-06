import pandas as pd
import sqlite3
from b_function_visualization import *
from c_chatbot import *

# df = pd.read_csv('전처리_내역.csv')
df1 = pd.read_excel('sampling1.xlsx')
df2 = pd.read_excel('sampling2.xlsx')
df3 = pd.read_excel('sampling3.xlsx')

# 데이터베이스 연결 및 테이블 생성
def init_db():
    conn = sqlite3.connect('user_transactions.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_metadata (
        username TEXT,
        account_number TEXT,
        table_name TEXT,
        PRIMARY KEY (username, account_number)
    )
    ''')
    conn.commit()
    return conn


# 사용자 데이터 불러오기
def load_user_data(conn, username, account_number):
    cursor = conn.cursor()
    cursor.execute('''
    SELECT table_name FROM user_metadata
    WHERE username = ? AND account_number = ?
    ''', (username, account_number))
    result = cursor.fetchone()
    
    if result:
        table_name = result[0]
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    else:
        return None


def login(username, account_number, selected_year):
    global global_user_data  # 전역 변수를 사용
    conn = init_db()  # 데이터베이스 초기화
    global_user_data = load_user_data(conn, username, account_number)  # 사용자 데이터 불러오기

    if global_user_data is not None:
        # 시각화 및 기타 작업
        top_categories, character = top_k_category(global_user_data, 5)
        top_category_img = plot_top_category(global_user_data, 5)  # base64 이미지 반환
        monthly_consumption_data = calculate_monthly_consumption(global_user_data, selected_year)
        monthly_consumption_img = plot_monthly_consumption(global_user_data, selected_year)  # base64 이미지 반환
        card = card_recommendation()

        # 데이터베이스 연결 종료
        conn.close()

        # 데이터 및 기타 반환 값
        return (
            global_user_data,  # DataFrame 반환
            "로그인 성공! 환영합니다!",
            gr.Image(value=top_category_img, visible=True),  # base64 이미지로 업데이트
            gr.update(visible=True),  # 연도 선택 라디오 버튼 표시
            gr.Image(value=monthly_consumption_img, visible=True),  # 월별 시각화 base64 이미지로 업데이트
            gr.update(value=character, visible=True),  # 캐릭터 출력
            gr.update(value=card, visible=True),  # 카드 추천 출력
            gr.update(visible=True),  # 챗봇 활성화
            gr.update(visible=False)  # 로그인 폼 숨기기
        )
    else:
        conn.close()  # 데이터베이스 연결 종료

        # 사용자 데이터가 없을 때 처리
        return (
            None,  # user_data 없음
            f"{username}님의 거래 내역을 찾을 수 없습니다.",
            gr.update(visible=False),  # 지출 top 시각화 숨김
            gr.update(visible=False),  # 연도 선택 라디오 버튼 숨김
            gr.update(visible=False),  # 월별 시각화 숨김
            gr.update(visible=False),  # 캐릭터 숨김
            gr.update(visible=False),  # 카드 추천 숨김
            gr.update(visible=False),  # 챗봇 숨김
            gr.update(visible=True)  # 로그인 폼 표시
        )

# 연도 선택 라디오 버튼이 변경되면 월별 소비내역 시각화 업데이트
def update_monthly_consumption(user_data, selected_year):
    if user_data is not None:
        # 월별 소비 내역 시각화
        monthly_consumption_img = plot_monthly_consumption(user_data, selected_year)
        return gr.update(value=monthly_consumption_img, visible=True)
    else:
        return gr.update(visible=False)



# 사용자 데이터 저장
def save_user_data(conn, username, account_number, df):
    cursor = conn.cursor()
    table_name = f"transactions_{username}_{account_number}"
    
    # 메타데이터 저장
    cursor.execute('''
    INSERT OR REPLACE INTO user_metadata (username, account_number, table_name)
    VALUES (?, ?, ?)
    ''', (username, account_number, table_name))
    
    # 트랜잭션 데이터 저장
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()


# 데이터 저장
def save_data():
    conn = init_db()
    save_user_data(conn, "한윤지", "A001", df1)
    save_user_data(conn, "손아현", "B002", df2)
    save_user_data(conn, "신지후", "C003", df3)
    
    conn.close()


# 데이터 저장
save_data()

# 외부 모듈에서 데이터프레임 접근 방법
def get_user_data():
    global global_user_data
    return global_user_data




#저장 데이터 출력해보기 -- main 에서만 할 듯
def query_user_transactions(username, account_number, db_path='user_transactions.db'):
    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 사용자의 거래 내역 테이블 이름 조회
        cursor.execute('''
        SELECT table_name FROM user_metadata
        WHERE username = ? AND account_number = ?
        ''', (username, account_number))
        
        result = cursor.fetchone()
        
        if result:
            table_name = result[0]
            # 거래 내역 조회
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            return df
        else:
            return None
    finally:
        conn.close()

# 메타데이터 테이블 전체를 조회
def check_user_metadata():
    # Connect to the database
    conn = sqlite3.connect('user_transactions.db')
    cursor = conn.cursor()
    
    # Query the user_metadata table
    query = "SELECT * FROM user_metadata"
    cursor.execute(query)
    result = cursor.fetchall()
    
    # Convert result to DataFrame for better readability
    df = pd.DataFrame(result, columns=["username", "account_number", "table_name"])
    
    # Close the connection
    conn.close()
    
    return df

if __name__ == "__main__":
    # 사용 예시
    username = "한윤지"  # 조회할 사용자 이름
    account_number = "A001"  # 조회할 계좌번호

    transactions = query_user_transactions(username, account_number)

    if transactions is not None:
        print(f"{username}님의 거래 내역:")
        print(transactions)
    else:
        print(f"{username}님의 거래 내역을 찾을 수 없습니다.")

    login('손아현', 'B002')
    # check_user_metadata()
