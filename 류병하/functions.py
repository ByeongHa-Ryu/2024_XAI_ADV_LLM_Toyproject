import pandas as pd 
from datetime import datetime
from dateutil.relativedelta import relativedelta



""" Functions for Auto Analysis """

# Return Top k categories & Search Corresponding Character

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



def montly_consumption(df,year):
    filtered_df = df[df['연도']==year]
    montly_consumption = filtered_df.groupby('월')['출금금액'].agg('sum').reset_index()
    return montly_consumption


