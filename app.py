import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from textblob import TextBlob
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 引入 StandardScaler
import nltk
import re
import matplotlib.pyplot as plt 
# 引入 PyTorch 核心模組
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 下載必要的 NLTK 數據
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 頁面設定 ---
st.set_page_config(page_title="YouTube 情感分析與 LSTM 流量預測系統", layout="wide")

st.title("📊 YouTube 情感分析與 LSTM 流量預測系統")
# st.markdown("""
# 本系統整合 **NLP 情感分析**與 **PyTorch LSTM** 模型。
# **預測設定：** 使用所有歷史數據進行訓練，並固定預測**未來 30 天**的趨勢。
# """)

# --- 側邊欄：設定 ---
st.sidebar.header("⚙️ 參數設定")
api_key = st.sidebar.text_input("輸入 YouTube Data API Key", type="password")
video_input = st.sidebar.text_input("輸入 YouTube 影片 ID 或網址", value="fB8TyLTD7EE")
max_results = st.sidebar.slider("抓取評論數量上限 (最大 50000 筆)", 100, 50000, 50000)
# 固定預測期為 30 天
FORECAST_PERIOD = 30 
LOOK_BACK = 7 # LSTM Lookback

# --- 函數定義區 ---

def extract_video_id(input_str):
    """從網址或髒亂的字串中提取純淨的 Video ID"""
    if not input_str: return ""
    match_standard = re.search(r'v=([a-zA-Z0-9_-]{11})', input_str)
    if match_standard: return match_standard.group(1)
    match_short = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', input_str)
    if match_short: return match_short.group(1)
    if '?' in input_str: return input_str.split('?')[0]
    return input_str.strip()

@st.cache_data(ttl=3600) 
def get_video_comments(api_key, video_id, max_results):
    """資料蒐集：透過 YouTube API 抓取評論 (內容與原版相同)"""
    if not api_key or not video_id: return pd.DataFrame()
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_data = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
        )
        
        while request and len(comments_data) < max_results:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'text': comment['textDisplay'],
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt'],
                    'author': comment['authorDisplayName']
                })
            
            if 'nextPageToken' in response and len(comments_data) < max_results:
                request = youtube.commentThreads().list(
                    part="snippet", videoId=video_id, maxResults=100,
                    textFormat="plainText", pageToken=response['nextPageToken']
                )
            else:
                break
    except Exception as e:
        st.error(f"API 抓取錯誤: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(comments_data)

def analyze_sentiment(text):
    """情感分析 (Demo 使用 TextBlob)"""
    return TextBlob(str(text)).sentiment.polarity

def extract_topics(texts, n_topics=5):
    """主題建模 (使用 NMF)"""
    if len(texts) < 5: return ["樣本過少，無法提取主題"]
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        X = tfidf.fit_transform(texts)
        n_components = min(n_topics, X.shape[0], X.shape[1])
        if n_components < 2: return ["文本內容不足以提取關鍵字"]
        nmf = NMF(n_components=n_components, random_state=42)
        nmf.fit(X)
        keywords = []
        feature_names = tfidf.get_feature_names_out()
        for topic_idx, topic in enumerate(nmf.components_):
            top_features_ind = topic.argsort()[:-6:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            keywords.append(f"主題 {topic_idx+1}: {', '.join(top_features)}")
        return keywords
    except ValueError:
        return ["文本內容不足以提取關鍵字"]

# --- PyTorch LSTM 模型定義 (內容與原版相同) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_lstm_dataset(data, look_back=7, is_forecast=False):
    """將時間序列數據轉換為 LSTM 模型所需的輸入格式並進行縮放 (內容與原版相同)"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['y'].values.reshape(-1, 1))
    
    if is_forecast:
        X = scaled_data[-look_back:, 0]
        return X, scaler

    X, Y = [], []
    for i in range(len(scaled_data) - look_back):
        a = scaled_data[i:(i + look_back), 0]
        X.append(a)
        Y.append(scaled_data[i + look_back, 0])
        
    return np.array(X), np.array(Y), scaler

# --- 繪圖函數 (內容與原版相同) ---

def plot_lstm_results(train_dates, forecast_dates, actual_values, predictions, title):
    """繪製 LSTM 預測結果 (使用所有數據訓練，預測未來 N 天)"""
    fig = go.Figure()
    
    # 訓練數據 (所有歷史實際數據)
    fig.add_trace(go.Scatter(x=train_dates, y=actual_values, 
                             mode='lines+markers', name='歷史數據 (用於訓練)', 
                             line=dict(color='gray', width=1.5), marker=dict(size=4)))
    
    # 預測線
    fig.add_trace(go.Scatter(x=forecast_dates, y=predictions, 
                             mode='lines+markers', name=f'LSTM 預測 (未來{len(forecast_dates)}天)', 
                             line=dict(color='purple', width=3), marker=dict(size=6)))
    
    fig.update_layout(title=title,
                      xaxis_title='日期', yaxis_title='每日評論量',
                      hovermode='x unified')
    return fig

# --- 主程式邏輯 ---

if st.sidebar.button("開始分析流程"):
    if not api_key:
        st.error("請輸入 API Key！")
    elif not video_input:
        st.error("請輸入影片 ID！")
    else:
        # 自動清洗 Video ID
        clean_video_id = extract_video_id(video_input)
        st.sidebar.success(f"已識別影片 ID: {clean_video_id}")

        # [修改] 建立 Tabs
        tab1, tab2, tab3 = st.tabs(["1. 資料蒐集 & 前處理", "2. NLP 情感與主題", f"3. PyTorch LSTM 預測 (未來{FORECAST_PERIOD}天)"])

        # --- 階段 1: 資料蒐集 ---
        with tab1:
            st.subheader("📥 資料蒐集 (Data Collection)")
            st.markdown(f"目標抓取上限：**{max_results} 筆評論**，以獲取最長歷史數據。")
            with st.spinner(f"正在從 YouTube 抓取資料 (ID: {clean_video_id})..."):
                df = get_video_comments(api_key, clean_video_id, max_results)
            
            if not df.empty:
                # 移除時區資訊
                df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
                st.success(f"成功抓取 {len(df)} 筆評論！")
                
                # 顯示數據時間跨度
                min_date = df['published_at'].min().strftime('%Y-%m-%d')
                max_date = df['published_at'].max().strftime('%Y-%m-%d')
                time_span = (df['published_at'].max().normalize() - df['published_at'].min().normalize()).days
                
                st.info(f"**數據時間範圍：** 從 **{min_date}** 到 **{max_date}**，共覆蓋約 **{time_span} 天** (不連續)。")
                st.dataframe(df.head())
                
                # 時間分佈圖
                fig_hist = px.histogram(df, x="published_at", title="評論發佈時間分佈")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.error("無法獲取數據，請檢查 API Key 或影片 ID 是否正確。")
                st.stop()

        # --- 階段 2: NLP 分析 ---
        with tab2:
            st.subheader("🧠 情感分析與主題建模 (NLP)")
            
            with st.spinner("正在進行情感運算..."):
                # 情感分析
                df['sentiment'] = df['text'].apply(analyze_sentiment)
                df['sentiment_label'] = df['sentiment'].apply(lambda x: '正面' if x > 0.05 else ('負面' if x < -0.05 else '中立'))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 情感分佈")
                    fig_pie = px.pie(df, names='sentiment_label', title='評論情感佔比', color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.markdown("#### 主題關鍵字 (Topic Modeling)")
                    topics = extract_topics(df['text'].dropna())
                    for t in topics:
                        st.write(f"- {t}")
                
                st.markdown("---")
                st.markdown("**情感隨時間變化趨勢**")
                # 按天聚合
                df['date'] = df['published_at'].dt.date
                daily_data = df.groupby('date').agg({
                    'sentiment': 'mean',
                    'text': 'count'
                }).reset_index().rename(columns={'text': 'volume'})
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                
                # 繪製每日情感趨勢圖
                fig_trend = px.line(daily_data, x='date', y='sentiment', title='每日平均情感分數趨勢')
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # =======================================================
                # [新增] 評論數量與情感分數的關係圖表
                # =======================================================
                st.markdown("### 🔗 評論數量 (流量) 與情感分數的關係")
                
                if len(daily_data) >= 20: # 確保至少有 20 天數據再繪製複雜關係圖
                    
                    # 1. 計算相關係數
                    correlation = daily_data['sentiment'].corr(daily_data['volume'])
                    st.info(f"觀看量替代指標 (評論數) 與情感分數的相關係數 r = {correlation:.3f}")
                    
                    # 2. 繪製散點圖 (用於判斷線性關係)
                    fig_scatter = px.scatter(daily_data, x='sentiment', y='volume', 
                                             title=f'情感分數 vs. 評論數量散點圖 (r={correlation:.3f})',
                                             labels={'sentiment': '平均情感分數', 'volume': '每日評論數量'})
                    fig_scatter.update_layout(xaxis_range=[-1, 1]) # 限制情感分數範圍在 -1 到 1
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # 3. 繪製雙軸時間趨勢圖 (用於判斷滯後性)
                    
                    # 標準化數據 (MinMaxScaler)
                    scaler_volume = MinMaxScaler()
                    scaler_sentiment = MinMaxScaler()
                    
                    daily_data['volume_norm'] = scaler_volume.fit_transform(daily_data[['volume']])
                    daily_data['sentiment_norm'] = scaler_sentiment.fit_transform(daily_data[['sentiment']])
                    
                    fig_dual_norm = go.Figure()
                    
                    # 繪製標準化評論數量 (左軸)
                    fig_dual_norm.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['volume_norm'], 
                                                       name='評論數量 (標準化)', line=dict(color='blue', width=2)))
                    
                    # 繪製標準化情感分數 (右軸)
                    fig_dual_norm.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['sentiment_norm'], 
                                                       name='情感分數 (標準化)', line=dict(color='red', width=2, dash='dot')))
                    
                    fig_dual_norm.update_layout(title='標準化評論數量與情感分數趨勢對比 (判斷滯後性)',
                                                xaxis_title='日期',
                                                yaxis_title='標準化數值',
                                                hovermode='x unified')
                    st.plotly_chart(fig_dual_norm, use_container_width=True)
                
                else:
                    st.warning("數據點不足 20 天，無法進行有效的關係分析。")

        # --- 數據準備 (供 LSTM 使用) ---
        LOOK_BACK = 7 # LSTM Lookback
        # 檢查數據點是否足夠：Lookback + 預測期
        if len(daily_data) < FORECAST_PERIOD + LOOK_BACK:
            st.warning(f"數據點過少 (至少需要 {FORECAST_PERIOD + LOOK_BACK} 天)，無法進行 {FORECAST_PERIOD} 天的預測。", icon="⚠️")
            prophet_df = None
        else:
            # 使用所有歷史數據進行訓練
            prophet_df = daily_data[['date', 'volume']].rename(columns={'date': 'ds', 'volume': 'y'}).reset_index(drop=True)
            
            st.sidebar.markdown("---")
            st.sidebar.info(f"**歷史數據天數：** {len(prophet_df)} 天")
            st.sidebar.info(f"**預測目標：** 未來 {FORECAST_PERIOD} 天")


        # --- 階段 3: PyTorch LSTM 預測 (固定預測未來 N 天) ---
        with tab3:
            st.subheader(f"🤖 PyTorch LSTM 預測 (訓練所有歷史數據，預測未來 {FORECAST_PERIOD} 天)")
            
            if prophet_df is not None:
                st.markdown("LSTM 網路將從所有歷史數據中學習時間模式，並使用滾動預測 (Rolling Forecast) 來推算未來趨勢。")
                
                INPUT_SIZE = 1       
                HIDDEN_SIZE = 50     
                NUM_LAYERS = 1       
                NUM_EPOCHS = 20      
                BATCH_SIZE = 1       
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                st.sidebar.info(f"PyTorch 使用設備: {device}")
                
                # 1. 訓練數據準備 (使用所有歷史數據)
                X_train_data, Y_train_data, scaler = create_lstm_dataset(prophet_df, look_back=LOOK_BACK, is_forecast=False)
                
                X_train = torch.tensor(X_train_data, dtype=torch.float32).unsqueeze(-1).to(device)
                Y_train = torch.tensor(Y_train_data, dtype=torch.float32).unsqueeze(-1).to(device)
                
                train_dataset = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                st.info(f"LSTM 訓練數據點：{len(X_train)} 筆 (用於預測基礎的歷史數據點)")
                
                with st.spinner(f"正在訓練 PyTorch LSTM 模型 (Epochs={NUM_EPOCHS})..."):
                    
                    # 2. 建立模型、損失函數和優化器
                    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    
                    # 3. 訓練迴圈
                    model.train()
                    for epoch in range(NUM_EPOCHS):
                        for i, (inputs, labels) in enumerate(train_loader):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                    # 4. 進行未來預測
                    model.eval() 
                    with torch.no_grad():
                        # 4a. 初始化預測輸入 (最後一個 look_back 序列)
                        last_sequence_np, _ = create_lstm_dataset(prophet_df, look_back=LOOK_BACK, is_forecast=True)
                        current_input = torch.tensor(last_sequence_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # Shape: (1, look_back, 1)
                        
                        forecast_predictions = []
                        
                        for _ in range(FORECAST_PERIOD):
                            # 預測下一個時間點
                            predicted_value_tensor = model(current_input) # Shape: (1, 1)
                            forecast_predictions.append(predicted_value_tensor.cpu().numpy()[0, 0])
                            
                            # 滾動預測：更新輸入序列
                            predicted_value_scaled = predicted_value_tensor.clone().detach() 
                            
                            # 移除第一個元素，並在末尾添加新預測值
                            new_input_np = current_input.cpu().numpy()[0, 1:, 0]
                            new_input_np = np.append(new_input_np, predicted_value_scaled.cpu().numpy()[0, 0])
                            
                            # 更新 current_input
                            current_input = torch.tensor(new_input_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                        
                    # 5. 反向縮放數據，還原為原始評論量
                    forecast_predictions = np.array(forecast_predictions).reshape(-1, 1)
                    
                    # 重新獲取 scaler，用於反向轉換
                    _, _, final_scaler = create_lstm_dataset(prophet_df, look_back=LOOK_BACK, is_forecast=False)
                    
                    final_predictions = final_scaler.inverse_transform(forecast_predictions).flatten()
                    
                    # 6. 繪圖準備：生成未來日期
                    last_date = prophet_df['ds'].max()
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_PERIOD)
                    
                    st.write(f"#### 🚀 未來 {FORECAST_PERIOD} 天預測結果")
                    forecast_table = pd.DataFrame({
                        '日期': forecast_dates,
                        '預測評論量': final_predictions.round(1).clip(min=0) # 評論量不能為負數
                    })
                    st.dataframe(forecast_table)

                    # 繪製結果
                    fig_lstm = plot_lstm_results(
                        train_dates=prophet_df['ds'],
                        forecast_dates=forecast_dates,
                        actual_values=prophet_df['y'].values,
                        predictions=final_predictions,
                        title=f'PyTorch LSTM 模型預測 (預測未來 {FORECAST_PERIOD} 天)'
                    )
                    st.plotly_chart(fig_lstm, use_container_width=True)
                    
            else:
                 pass

else:

    st.info("👈 請在側邊欄輸入資料並點擊按鈕，開始多模型分析。")
