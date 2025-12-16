# final_report
# 📊 YouTube 輿情分析與三模型流量預測系統


本專案是一個基於 **Streamlit** 的整合式數據分析平台，旨在分析 YouTube 影片的觀眾輿情（NLP），並利用三種不同的時間序列模型（**PyTorch LSTM**）來預測影片的流量趨勢（觀看次數）。

## 🚀 功能特色

本系統基於 **CRISP-DM** 標準流程開發，主要功能包含：

### 1. 資料蒐集與處理 (Data Collection)
* **YouTube API 整合**：自動抓取指定影片的評論數據。

### 2. NLP 輿情分析 (Sentiment Analysis & NLP)
* **情感分析**：使用 TextBlob 分析評論的情感極性（正面/負面/中立）。
* **主題建模**：使用 NMF (Non-negative Matrix Factorization) 提取評論中的關鍵主題。
* **關聯分析**：視覺化「情感分數」與「評論數量/觀看流量」之間的相關性與滯後效應。

### 3. 多模型流量預測 (Time Series Forecasting)
所有模型皆採用 **80% 訓練集 / 20% 測試集** 的嚴謹切分方式進行評估：
* **PyTorch LSTM**：基於深度學習的長短期記憶網路，捕捉非線性關係與長期時間依賴性。

## 🛠️ 安裝說明

### 前置需求
* Python 3.8 或以上版本
* Google Cloud Console 的 **YouTube Data API v3 Key**



### 2. 安裝依賴套件
建議建立一個虛擬環境 (Virtual Environment) 後執行：
```bash
pip install -r requirements.txt
```

```text
streamlit
pandas
numpy
plotly
matplotlib
google-api-python-client
textblob
scikit-learn
nltk
prophet
statsmodels
torch
```


## ▶️ 使用方法

### 1. 啟動應用程序
streamlit run app.py


### 2. 操作介面設定 (側邊欄)
1.  **輸入 API Key**：貼上您的 YouTube Data API Key。
2.  **輸入影片 ID**：貼上 YouTube 影片網址或 ID (用於抓取評論進行 NLP 分析)。

### 3. 開始分析
點擊側邊欄的 **「開始分析流程」** 按鈕，系統將自動執行並生成三個分頁報告。


## 🧠 模型架構細節

### PyTorch LSTM
* **架構**：單層 LSTM + 全連接層 (Linear)。
* **Input Size**: 1 (單變量時間序列)。
* **Hidden Size**: 50。
* **Lookback**: 7 (使用過去 7 天預測下一天)。
* **Optimizer**: Adam (Learning Rate = 0.001)。
* **Loss Function**: MSE (均方誤差)。


## 專案結構

.
├── app.py                  # 主程式碼 (Streamlit 應用入口)
├── requirements.txt        # 依賴套件清單
├── README.md               # 專案說明文件
└── data/                   # (可選) 存放範例 CSV 數據
