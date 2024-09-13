# 2024_XAI_ADV_LLM_Toyproject
국민대학교 인공지능 학회 X:AI 의 2024년 ADV session 에서 진행한 LLM 팀의 Toy Project 입니다. 

# <span style="color: orange"> 텅후루:
### <span style="color: navy"> 생성형 AI 기반 개인화 소비 분석 금융 챗봇</span>

## 프로젝트 개요
LLM을 활용하여 소비 내역을 분석하고 사용자 맞춤형 카드를 추천해주는 챗봇을 개발하는 것을 목표로 하고 있습니다. 사용자는 자신의 소비 데이터를 기반으로 실시간 분석 및 피드백을 받을 수 있으며, 또한 개인의 소비 패턴에 맞는 신용카드를 추천받을 수 있습니다.

## 팀원 소개
- 한윤지 : SQLite 데이터베이스, FAISS 데이터베이스, 카드 추천
- 류병하 : 카드 추천, 챗봇(LangChain), Pandas Agent 
- 손아현 : 데이터 크롤링, Gradio
- 신지후 : 데이터 수집, KoBERT 라벨링

## 서비스 소개
### 서비스 구조
![image](https://github.com/user-attachments/assets/f81f8bec-d1b7-4b6a-b891-9ca6f6b07369)

[ 기술 스택 ]

| KoBERT                | SQLite            | Faiss           | GPT          | Gradio         |
|-----------------------|-------------------|-----------------|--------------|----------------|
| 카테고리 분류 모델    | 거래 내역 DB      | 카드 정보 DB    | 챗봇 LLM     | 데모 UI 제작   |
### 주요 기능
1. 소비 내역 분석
사용자의 소비 내역을 분석하고, 소비 카테고리별로 분류하여 피드백을 제공합니다.
공차, 스타벅스 등 다양한 상호명을 카테고리화하여 정확한 소비 분석을 지원합니다.
2. 카드 추천 시스템
사용자의 소비 패턴을 바탕으로 최적의 카드를 추천합니다.
국민카드 등 다양한 카드사의 데이터를 기반으로 사용자 맞춤형 추천을 제공합니다.
3. 실시간 피드백 및 사용자 인터페이스
Gradio 인터페이스로 구현된 챗봇을 통해 실시간으로 소비 분석 및 카드를 추천받을 수 있습니다.
향후 사용자 친화적인 UI를 추가하여 보다 직관적인 사용 경험을 제공할 예정입니다.
![image](https://github.com/user-attachments/assets/57fd425e-6498-4f67-a29e-47017cd891ce)
![image](https://github.com/user-attachments/assets/c5056e5e-c76c-460b-98c0-0a87d28b26c0)

## 설계 과정
**1. 소비 내역 데이터 처리** </br>
 - 사용자의 소비 내역에서 상호명(e.g 공차, 스타벅스 등)을 분석하여, 상호명을 '카페', '음식점' 등의 카테고리로 분류하는 작업을 KoBERT 모델을 통해 수행합니다.
 - 분류된 소비 데이터는 사용자 메타데이터(사용자명, 계좌번호, 거래 내역 테이블 이름)를 저장하는 user_metadata 테이블과 함께 각 사용자별로 동적으로 생성된 테이블에 SQLite 데이터베이스에 저장됩니다.
 - 사용자가 로그인하면, SQLite 데이터베이스에서 해당 사용자의 소비 내역(카테고리화된 데이터 포함)을 불러옵니다.

**2. LLM 블록 시스템** </br>
 - LLM 블록 구조를 도입하여, 각 블록이 특정 작업을 처리하도록 설계되었습니다.
 - 예를 들어, 사용자의 입력에 따라 소비 내역 분석 블록, 카드 추천 블록 등이 개별적으로 작동하여 효율적인 데이터 처리를 수행합니다.
<br> **금융 데이터 분석** </br>
 - 금융 관련 데이터 분석을 위해 GPT-4 API와 프롬프트 엔지니어링을 활용하였습니다.
 - 데이터 프레임 분석 에이전트를 활용하여 소비 데이터를 판다스(Pandas) 기반으로 분석하고, 그 결과를 제공합니다.
 <br>**카드 추천 시스템** </br>
 - 카드 추천 시스템은 FAISS를 기반으로 구현되며, KoBERT 모델로 카테고리화된 소비 내역을 바탕으로 유사한 소비 패턴을 가진 카드를 추천합니다.
 - 국민카드 등의 카드 설명 pdf를 TDF 형식으로 저장하고, 리트리버 시스템을 통해 사용자 맞춤형 카드 추천을 진행합니다.
