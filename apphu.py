import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Stock Preprocessing App", layout="wide")
st.title("📈 Stock Analysis and Preprocessing App")
#L5X1DGIJ0OVOELK1

# Now continue analysis only if data is fetched
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "data_fetched" not in st.session_state:
    st.session_state.data_fetched = False

# Stock list
stock_list = {
    "Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Tesla": "TSLA", "Amazon": "AMZN",
    "NVIDIA": "NVDA", "Meta (Facebook)": "META", "Netflix": "NFLX", "Intel": "INTC", "AMD": "AMD",
    "Reliance": "RELIANCE.NS", "Infosys": "INFY.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "SBI": "SBIN.NS", "ICICI Bank": "ICICIBANK.NS", "Axis Bank": "AXISBANK.NS", "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "Adani Enterprises": "ADANIENT.NS", "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "Maruti Suzuki": "MARUTI.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "HCL Tech": "HCLTECH.NS",
    # Added US
    "PayPal": "PYPL", "Salesforce": "CRM", "Adobe": "ADBE", "Qualcomm": "QCOM",
    "PepsiCo": "PEP", "Coca‑Cola": "KO", "McDonald’s": "MCD", "Boeing": "BA",
    "Ford": "F", "Walmart": "WMT", "Visa": "V", "Mastercard": "MA",
    "JPMorgan": "JPM", "Goldman Sachs": "GS", "ExxonMobil": "XOM",
    # Added India
    "LTIMindtree": "LTIM.NS", "Tata Motors": "TATAMOTORS.NS", "JSW Steel": "JSWSTEEL.NS",
    "UltraTech Cement": "ULTRACEMCO.NS", "Dr. Reddy’s": "DRREDDY.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Tech Mahindra": "TECHM.NS", "ONGC": "ONGC.NS", "Coal India": "COALINDIA.NS",
    "Havells": "HAVELLS.NS", "Zomato": "ZOMATO.NS", "Nykaa": "NYKAA.NS",
    "DMart": "DMART.NS", "Bharat Electronics": "BEL.NS", "IndusInd Bank": "INDUSINDBK.NS"
}

company = st.selectbox("Select a Company:", list(stock_list.keys()))
ticker = stock_list[company]
min_date=datetime.date(2000,1,1)
max_date=datetime.date.today()
default_start=datetime.date(2010,1,1)
default_end=datetime.date(2020,1,1)

start_date = st.date_input("Start date",value=default_start,min_value=min_date,max_value=max_date)
end_date = st.date_input("End date",value=default_end,min_value=min_date,max_value=max_date)
#start_date = st.date_input("Start Date", value=datetime.date(2010, 1, 1), min_value=datetime.date(1990, 1, 1))
#end_date = st.date_input("End Date", value=datetime.date.today())

if st.button("Fetch Stock Data"):
    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        with st.spinner("Fetching data..."):
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning("No data found for the selected range.")
                st.session_state.stock_data = None
                st.session_state.data_fetched = False
            else:
                st.session_state.stock_data = data
                st.session_state.data_fetched = True
                st.success("Data fetched successfully!")

# Now continue analysis only if data is fetched
if st.session_state.data_fetched:
    data = st.session_state.stock_data.copy()

    st.subheader("📊 First 5 Rows of the Dataset")
    st.dataframe(data.head())

    # Null check and handling
    st.subheader("🧹 Null Value Check and Handling")
    st.write("Null values before fillna:", data.isnull().sum())
    data.fillna(method='ffill', inplace=True)
    st.write("Null values after fillna:", data.isnull().sum())

    # ADF test
    st.subheader("🧪 ADF Test for Stationarity")
    result = adfuller(data['Close'].values)
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])
    if result[1] < 0.05:
        st.success("✅ Reject Null Hypothesis — Data is Stationary.")
    else:
        st.warning("⚠ Fail to Reject Null Hypothesis — Data is Non-Stationary.")

    # Seasonality and Trend
    st.subheader("📉 Seasonality, Trend and Residual Plot")
    data_monthly = data.resample('M').sum()
    decomp = seasonal_decompose(data_monthly['Close'], model='additive')

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    decomp.observed.plot(ax=axs[0], title='Observed')
    decomp.trend.plot(ax=axs[1], title='Trend')
    decomp.seasonal.plot(ax=axs[2], title='Seasonality')
    decomp.resid.plot(ax=axs[3], title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

    # Volatility plot
    st.subheader("⚡ Volatility Check: Daily Returns")
    data['Returns'] = data['Close'].pct_change()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(data['Returns'], color='orange')
    ax2.set_title('Daily Returns')
    st.pyplot(fig2)
    import pandas as pd
    import numpy as np
    import pyttsx3
    import warnings
    import streamlit as st
    from statsmodels.tsa.stattools import adfuller, acf
    from statsmodels.tsa.seasonal import seasonal_decompose

    warnings.filterwarnings("ignore")


    # --------- Optional Text-to-Speech ---------
    def speak(text):
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()


    # --------- Utilities ---------
    def has_missing(ts):
        return ts.isnull().any().item()  # ✅ Safe and clean using .any().item()


    def is_stationary(ts):
        result = adfuller(ts.dropna())
        return result[1] < 0.05


    def has_trend(ts):
        try:
            result = seasonal_decompose(ts, period=12, model='additive')
            return result.trend.dropna().std() > 0.1
        except:
            return False


    def has_seasonality(ts):
        try:
            result = seasonal_decompose(ts, period=12, model='additive')
            return result.seasonal.dropna().std() > 0.05
        except:
            return False


    def is_nonlinear(ts):
        try:
            acfs = acf(ts.dropna(), nlags=5)
            return abs(acfs[1]) < 0.3
        except:
            return False


    def is_short_series(ts):
        return len(ts.dropna()) < 100


    # --------- Core Logic ---------
    def recommend_models(ts, speak_summary=False):
        ts = ts.squeeze()  # ensures it's a Series

        stationary = is_stationary(ts)
        trend = has_trend(ts)
        seasonal = has_seasonality(ts)
        nonlinear = is_nonlinear(ts)
        missing = has_missing(ts)
        short_series = is_short_series(ts)

        recs = []

        if stationary:
            recs.append("ARIMA")
        if trend or seasonal:
            recs.append("SARIMAX or Prophet")
        if nonlinear:
            recs += ["RandomForest", "SVR", "LSTM/GRU (future)"]
        if short_series:
            recs.append("Prophet or Tree-based Models")
        if missing:
            recs.append("RandomForest (handles missing)")

        recs = sorted(set(recs))

        summary = f"""
    🧾 DATA SUMMARY
    --------------------------
    Total data points: {len(ts)}
    Stationary: {"Yes" if stationary else "No"}
    Trend detected: {"Yes" if trend else "No"}
    Seasonality: {"Yes" if seasonal else "No"}
    Non-linear: {"Yes" if nonlinear else "No"}
    Missing values: {"Yes" if missing else "No"}
    Short series: {"Yes" if short_series else "No"}

    🤖 Recommended Forecasting Models:
    {', '.join(recs)}
    """

        st.text_area("📋 Model Recommendation Summary", summary.strip(), height=240)

        if speak_summary:
            speak(summary)

        return recs
    # --------- Streamlit Interface ---------
    #st.set_page_config(page_title="AI Forecast Model Recommender", layout="center
    st.title("📊 AI Forecast Model Recommender")
    df=data.copy()
    if 'Close' not in df.columns:
        st.error("❌ The file must contain a 'close' column.")
    else:
        ts = df['Close']

        st.line_chart(ts)

        speak_option = st.checkbox("🔊 Enable AI voice summary (optional)")

        if st.button("🔍 Analyze & Recommend Forecasting Models"):
            recommend_models(ts, speak_summary=speak_option)





    # ----------------- 🔍 Optional View: Fundamental Analysis -----------------
    st.subheader("📌 Want to view fundamental metrics?")
    fundamental_choice = st.selectbox("Select 1 to show the fundamental analysis table:", options=[0, 1],
                                      key="fundamental_choice")
    if fundamental_choice == 1:
        st.subheader("📊 Fundamental Analysis Metrics")

        # Get the fundamental data using yfinance.Ticker
        stock_obj = yf.Ticker(ticker)
        info = stock_obj.info

        # Create the fundamental metrics dictionary
        fundamentals = {
            "Metric": [
                "P/E Ratio", "P/B Ratio", "EPS", "ROE", "Profit Margin",
                "Debt to Equity", "Market Cap", "Dividend Yield",
                "Book Value", "Total Revenue"
            ],
            "Value": [
                info.get("trailingPE", "N/A"),
                info.get("priceToBook", "N/A"),
                info.get("trailingEps", "N/A"),
                info.get("returnOnEquity", "N/A"),
                info.get("profitMargins", "N/A"),
                info.get("debtToEquity", "N/A"),
                info.get("marketCap", "N/A"),
                info.get("dividendYield", "N/A"),
                info.get("bookValue", "N/A"),
                info.get("totalRevenue", "N/A")
            ]
        }

        descriptions = {
            "P/E Ratio": "Price-to-Earnings (Valuation)",
            "P/B Ratio": "Price-to-Book (Asset Valuation)",
            "EPS": "Earnings Per Share",
            "ROE": "Return on Equity (%)",
            "Profit Margin": "Net Profit % of Revenue",
            "Debt to Equity": "Financial Leverage",
            "Market Cap": "Total Market Value of the Company",
            "Dividend Yield": "Return via Dividends (%)",
            "Book Value": "Net Asset Value per Share",
            "Total Revenue": "Gross Earnings"
        }

        fundamentals_df = pd.DataFrame(fundamentals)
        fundamentals_df["Description"] = fundamentals_df["Metric"].map(descriptions)

        st.dataframe(fundamentals_df)
        with st.expander("📘 Click to View Interpretation of Each Metric"):
            st.markdown("""
            ### 🧾 Fundamental Metrics Explained:

            - *📊 P/E Ratio (Price-to-Earnings)*  
              - 🟢 < 10: Possibly undervalued  
              - 🔵 10–25: Fairly valued  
              - 🔴 > 30: Possibly overvalued  

            - *📘 P/B Ratio (Price-to-Book)*  
              - 🟢 < 1: Undervalued (trading below book value)  
              - 🔵 1–3: Reasonable  
              - 🔴 > 3: May be overvalued  

            - *💰 EPS (Earnings Per Share)*  
              - 🔴 < 0: Negative earnings  
              - 🔵 0–5: Low to moderate earnings  
              - 🟢 > 5: Strong profitability  

            - *📈 ROE (Return on Equity)*  
              - 🔴 < 5%: Weak  
              - 🔵 5–15%: Average  
              - 🟢 > 15%: Strong  

            - *💵 Profit Margin*  
              - 🔴 < 5%: Low profitability  
              - 🔵 5–20%: Acceptable  
              - 🟢 > 20%: Excellent  

            - *⚖ Debt to Equity Ratio*  
              - 🟢 < 0.5: Low risk  
              - 🔵 0.5–1: Manageable  
              - 🔴 > 1: High leverage (riskier)  

            - *🏢 Market Capitalization*  
              - 🟢 > ₹50,000 Cr / $10B: Large-cap (Stable)  
              - 🔵 ₹10,000–50,000 Cr / $2B–10B: Mid-cap  
              - 🔴 < ₹10,000 Cr / $2B: Small-cap (Volatile)  

            - *💸 Dividend Yield*  
              - 🔴 < 1%: Low (growth-focused)  
              - 🔵 1–4%: Balanced  
              - 🟢 > 4%: Attractive for income investors  

            - *📚 Book Value*  
              - ℹ Indicates value of company’s assets per share.  
              - 🟢 Higher = stronger fundamentals  

            - *💼 Total Revenue*  
              - ℹ Reflects scale and business size.  
              - 🟢 Rising trend = business expansion  
            """)
    else:
        st.info("Fundamental metrics table is hidden. Select 1 to view.")
    st.subheader("📌 Want to view model recommendations?")
    choice = st.selectbox("Select 1 to show the model recommendation table:", options=[0, 1])
    if choice == 1:
        st.subheader("📊 Model Recommendation Table Based on Data Characteristics")
        st.markdown("""
        | Data Characteristic                | Recommended Models                                                                 |
        |-----------------------------------|-------------------------------------------------------------------------------------|
        | 📉 *Stationary*                 | ARIMA, SARIMA, Holt-Winters                                                        |
        | 🔄 *Seasonal Patterns*          | SARIMA, Prophet, Holt-Winters,Garch(1,1) Model                                                    |
        | 📈 *Trend Present*              | Holt’s Linear Trend, Prophet, SARIMA,Garch(1,1) Model                                              |
        | ⚡ *Volatility / Variance Shift*| GARCH, XGBoost, LSTM                                                               |
        | 🧠 *Non-linear / Complex Patterns* | LSTM, GRU, XGBoost, DeepAR,RandomForest                                                        |
        | 🔁 *Multiple Variables Involved*| VAR, SARIMAX, XGBoost                                                              |
        """)
    else:
        st.info("Table hidden. Select 1 to view the recommendation table.")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,LSTM
    from sklearn.neural_network import MLPRegressor
    # Dataset already available
    df = data.copy()
    df['ds'] = df.index
    df['y'] = df['Close']


    # Forecast helper function
    def plot_forecast(original_df, forecast_df, title):
        plt.figure(figsize=(14, 6))
        plt.plot(original_df.index, original_df['Close'], label='Historical')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
        plt.axvline(original_df.index[-1], color='gray', linestyle='--', alpha=0.7)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)


    # Feature engineering for ML models
    def create_features(df):
        df_feat = df.copy()
        df_feat['lag1'] = df_feat['Close'].shift(1)
        df_feat['lag2'] = df_feat['Close'].shift(2)
        df_feat['lag3'] = df_feat['Close'].shift(3)
        df_feat['dayofweek'] = df_feat.index.dayofweek
        df_feat['month'] = df_feat.index.month
        return df_feat.dropna()


    # Model selector
    model = st.selectbox("Select Forecasting Model",
                         ["ARIMA", "SARIMAX", "Prophet", "Holt-Winters", "Random Forest", "HistGradientBoosting",
                          "SVR","LSTM"])

    if st.button("Generate Forecast"):
        future_days = 252
        future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

        if model == "ARIMA":
            arima_model = ARIMA(data['Close'], order=(5, 1, 0))
            arima_fit = arima_model.fit()
            forecast = arima_fit.forecast(steps=future_days)
            forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_index)
            plot_forecast(data, forecast_df, "ARIMA Forecast")

        elif model == "SARIMAX":
            sarimax_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarimax_fit = sarimax_model.fit(disp=False)
            forecast = sarimax_fit.forecast(steps=future_days)
            forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_index)
            plot_forecast(data, forecast_df, "SARIMAX Forecast")

        elif model == "Prophet":
            m = Prophet()
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=future_days)
            forecast = m.predict(future)
            forecast_df = forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'Forecast'})
            forecast_df = forecast_df[forecast_df.index > df.index[-1]]
            plot_forecast(df.set_index('ds'), forecast_df, "Prophet Forecast")

        elif model == "Holt-Winters":
            hw_model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
            hw_fit = hw_model.fit()
            forecast = hw_fit.forecast(steps=future_days)
            forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=future_index)
            plot_forecast(data, forecast_df, "Holt-Winters Forecast")

        elif model == "Random Forest":
            df_feat = create_features(data)
            X = df_feat[['lag1', 'lag2', 'lag3', 'dayofweek', 'month']]
            y = df_feat['Close']
            rf_model = RandomForestRegressor()
            rf_model.fit(X, y)

            last_known = df_feat.iloc[-3:].copy()
            preds = []

            for date in future_index:
                features = {
                    'lag1': last_known.iloc[-1]['Close'],
                    'lag2': last_known.iloc[-2]['Close'],
                    'lag3': last_known.iloc[-3]['Close'],
                    'dayofweek': date.dayofweek,
                    'month': date.month
                }
                X_pred = pd.DataFrame([features])
                pred = rf_model.predict(X_pred)[0]
                preds.append(pred)

                # Update last_known to maintain lag logic
                new_row = pd.DataFrame({'Close': [pred]}, index=[date])
                last_known = pd.concat([last_known, new_row])
                last_known = last_known.iloc[-3:]

            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "Random Forest Forecast")

        elif model == "HistGradientBoosting":
            df_feat = create_features(data)
            X = df_feat[['lag1', 'lag2', 'lag3', 'dayofweek', 'month']]
            y = df_feat['Close']
            hgb_model = HistGradientBoostingRegressor()
            hgb_model.fit(X, y)

            last_known = df_feat.iloc[-3:].copy()
            preds = []

            for date in future_index:
                features = {
                    'lag1': last_known.iloc[-1]['Close'],
                    'lag2': last_known.iloc[-2]['Close'],
                    'lag3': last_known.iloc[-3]['Close'],
                    'dayofweek': date.dayofweek,
                    'month': date.month
                }
                X_pred = pd.DataFrame([features])
                pred = hgb_model.predict(X_pred)[0]
                preds.append(pred)

                new_row = pd.DataFrame({'Close': [pred]}, index=[date])
                last_known = pd.concat([last_known, new_row])
                last_known = last_known.iloc[-3:]

            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "HistGradientBoosting Forecast")

        elif model == "SVR":
            df_feat = create_features(data)
            X = df_feat[['lag1', 'lag2', 'lag3', 'dayofweek', 'month']]
            y = df_feat['Close']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            svr_model = SVR(kernel='rbf')
            svr_model.fit(X_scaled, y)

            last_known = df_feat[['Close']].copy()
            lag_values = list(last_known['Close'].iloc[-3:].values)

            preds = []

            for date in future_index:
                features = {
                    'lag1': lag_values[-1],
                    'lag2': lag_values[-2],
                    'lag3': lag_values[-3],
                    'dayofweek': date.dayofweek,
                    'month': date.month
                }
                X_pred = pd.DataFrame([features])
                X_pred_scaled = scaler.transform(X_pred)
                pred = svr_model.predict(X_pred_scaled)[0]
                preds.append(pred)

                lag_values.append(pred)
                lag_values = lag_values[-3:]

            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "SVR Forecast")
        #elif model == "ANN":

        elif model == "LSTM":
            df_lstm = data.copy()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_lstm[['Close']])

            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            last_seq = scaled_data[-sequence_length:]
            preds = []
            for _ in range(future_days):
                input_seq = last_seq[-sequence_length:].reshape(1, sequence_length, 1)
                pred = lstm_model.predict(input_seq, verbose=0)[0][0]
                preds.append(pred)
                last_seq = np.append(last_seq, [[pred]], axis=0)

            preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            forecast_df = pd.DataFrame({'Forecast': preds}, index=future_index)
            plot_forecast(data, forecast_df, "LSTM Forecast")
    st.subheader('Live News Sentiment And Ai agent Recommendation')
    choice1 = st.selectbox("Select 1 to show the model recommendation table:", options=[0, 1],key='choice1')
    if choice1 == 1:
        import streamlit as st
        import feedparser
        import transformers
        import random

        # ======================
        # 📘 Your Stock List
        # ======================
        '''stock_list = [
            "Reliance Industries", "TCS", "Infosys", "HDFC Bank", "ICICI Bank",
            "Wipro", "Hindustan Unilever", "Axis Bank", "Kotak Mahindra Bank",
            "Tata Motors", "Apple", "Amazon", "Microsoft", "Google", "Tesla"
        ]'''


        # ======================
        # 📰 Agent 1: Get News
        # ======================
        def get_latest_news(stock_name, max_results=5):
            query = stock_name.replace(" ", "+")
            feed_url = f"https://news.google.com/rss/search?q={query}+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(feed_url)
            news_list = []

            for entry in feed.entries[:max_results]:
                news_list.append({
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.published
                })
            return news_list


        # ================================
        # 😐 Agent 2: Sentiment Analyzer
        # ================================
        sentiment_model = transformers.pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


        def analyze_sentiments(news_list):
            results = []
            for item in news_list:
                sentiment = sentiment_model(item['title'])[0]
                item['sentiment'] = sentiment['label']
                item['score'] = sentiment['score']
                results.append(item)
            return results


        # =============================
        # 📈 Agent 3: Forecasting Agent
        # =============================
        def forecast_stock_price(stock_name, days=15):
            # Placeholder: You can plug in your ARIMA, SARIMAX, LSTM, etc.
            # For now, return a random predicted value
            return round(random.uniform(1000, 3000), 2)


        # ==========================
        # 📋 Agent 4: Summary Maker
        # ==========================
        def generate_summary(stock_name, sentiment_results, forecast_value):
            positive = sum(1 for x in sentiment_results if x['sentiment'] == 'POSITIVE')
            negative = sum(1 for x in sentiment_results if x['sentiment'] == 'NEGATIVE')
            neutral = sum(1 for x in sentiment_results if x['sentiment'] == 'NEUTRAL')

            summary = f"""📊 Summary for {stock_name}:
        - Recent News Headlines: {len(sentiment_results)} articles
        - Positive: {positive}, Negative: {negative}, Neutral: {neutral}
        - Model Forecast: ₹{forecast_value:.2f}

        """

            if positive > negative:
                summary += "👉 Outlook: Positive sentiment. May show upward movement."
            elif negative > positive:
                summary += "⚠ Outlook: Negative news dominance. Be cautious."
            else:
                summary += "🟡 Outlook: Mixed signals. Wait for confirmation."

            return summary


        # ======================
        # 🖥 Streamlit UI
        # ======================
        st.title("📈 Trading AI Agent Assistant")

        stock = st.selectbox("Choose a Stock", stock_list)

        if st.button("Run AI Agent"):
            with st.spinner("Fetching news..."):
                news = get_latest_news(stock)

            with st.spinner("Analyzing sentiment..."):
                news_with_sentiment = analyze_sentiments(news)

            with st.spinner("Running forecast model..."):
                forecast_value = forecast_stock_price(stock, days=15)

            with st.spinner("Generating summary..."):
                summary = generate_summary(stock, news_with_sentiment, forecast_value)

            st.subheader("📰 Latest News & Sentiment")
            for item in news_with_sentiment:
                st.markdown(f"- [{item['title']}]({item['link']}) → *{item['sentiment']}* ({item['score']:.2f})")

            st.subheader("📃 Summary")
            st.code(summary, language='markdown')
    st.title("📈 Low‑Cost Growth Stock Finder (India & U.S.& others)")
    choice2=st.selectbox("enter 1 to Extract ",options=[0,1],key='choice2')
    if choice2==1:
        import requests
        from scipy.stats import linregress

        # 1. UI: Category + Country
        st.subheader("📈 Low‑Cost Growth Stock Finder (India & U.S.)")

        country = st.selectbox("🇮🇳/🇺🇸 Country", ["India", "USA", "others"])
        category = st.selectbox("Price Range", [
            "C1: <₹15 or <$2",
            "C2: ₹15–₹100 or $2–$10",
            "C3: ₹100–₹600 or $10–$50"
        ])

        # 2. Ticker lists (Expanded)
        ind = {
            "C1": ["NHPC.NS", "GMRINFRA.NS", "IDFCFIRSTB.NS", "SUZLON.NS", "PNB.NS", "IRFC.NS", "IOB.NS", "YESBANK.NS",
                   "UCOBANK.NS", "SJVN.NS"],
            "C2": ["BANKBARODA.NS", "CANBK.NS", "TATAMOTORS.NS", "CIPLA.NS", "SUNPHARMA.NS", "SAIL.NS", "BHEL.NS",
                   "UNIONBANK.NS", "NMDC.NS", "HUDCO.NS"],
            "C3": ["TCS.NS", "INFY.NS", "TATASTEEL.NS", "IOC.NS", "HCLTECH.NS", "HDFCBANK.NS", "RELIANCE.NS", "ONGC.NS",
                   "BPCL.NS", "LT.NS"]
        }
        us = {
            "C1": ["AACG", "AAME", "ABAT", "ABOS", "ABPA", "ADGI", "ALZN", "AEMD", "AVGR", "AYTU"],
            "C2": ["F", "SIRI", "AFRM", "IAG", "MNMD", "OPEN", "BBBYQ", "BBBY", "JOBY", "MARA"],
            "C3": ["META", "AAPL", "AMZN", "MSFT", "GOOG", "NFLX", "PYPL", "SQ", "TSLA", "NVDA"]
        }
        others_c1 = [
            "ALBKF.PK",  # Allied Bank (Pakistan, OTC)
            "YGEHY",  # Yingli Green Energy (China, OTC)
            "GELYF",  # Geely Automobile (China, OTC)
            "DNZOY",  # Denso Corp (Japan, OTC)
            "SIEGY",  # Siemens AG (Germany, OTC, sometimes dips <$2 in ADR splits)
            "NSANY",  # Nissan Motor Co (Japan)
            "HTHIY",  # Hitachi Ltd (Japan)
            "CHL",  # China Mobile (ADR - OTC)
            "VJET",  # Voxeljet AG (Germany)
            "SNP",  # China Petroleum & Chemical Corp (ADR)
        ]
        others_c2 = [
            "CAJ",  # Canon Inc (Japan)
            "TM",  # Toyota Motor Corp (Japan)
            "BASFY",  # BASF SE (Germany)
            "TCEHY",  # Tencent Holdings (China OTC)
            "HMC",  # Honda Motor (Japan)
            "FCAU",  # Fiat Chrysler (Merged into Stellantis)
            "EVKIF",  # Evonik Industries (Germany OTC)
            "CICHY",  # China Construction Bank (ADR)
            "ZNH",  # China Southern Airlines
            "BYDDF",  # BYD Company (China, OTC)
        ]
        others_c3 = [
            "SONY",  # Sony Group Corp (Japan)
            "SAP",  # SAP SE (Germany)
            "BIDU",  # Baidu Inc (China)
            "JD",  # JD.com (China)
            "NTDOY",  # Nintendo Co Ltd (Japan)
            "VWAGY",  # Volkswagen AG (Germany)
            "NMR",  # Nomura Holdings (Japan)
            "YUMC",  # Yum China Holdings
            "BABA",  # Alibaba Group (China)
            "INFY",  # Infosys ADR (India)
        ]
        others = {"C1": others_c1, "C2": others_c2, "C3": others_c3}

        sel = category.split(":")[0]
        tickers = ind[sel] if country == "India" else us[sel] if country == "USA" else others[sel]

        # 3. Currency conversion
        try:
            usd_to_inr = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()["rates"]["INR"]
        except:
            usd_to_inr = 83.0  # fallback


        # 4. Fetching stock data
        @st.cache_data(show_spinner=False)
        def fetch(tk):
            try:
                o = yf.Ticker(tk)
                hist = o.history(period="1y")["Close"]
                if len(hist) < 60:
                    return None
                info = o.info
                cur = info.get("currency", "USD")
                price = hist.iloc[-1] * (usd_to_inr if cur == "USD" else 1)
                ret1y = hist.iloc[-1] / hist.iloc[0] - 1
                ret1m = hist.iloc[-1] / hist.iloc[-21] - 1 if len(hist) > 21 else np.nan
                slope = linregress(np.arange(len(hist)), hist.values).slope
                pe = info.get("trailingPE", np.nan)
                roe = (info.get("returnOnEquity", 0) or 0) * 100
                high52 = hist.max()
                new_growth = 1 if hist.iloc[-1] > 0.9 * high52 else 0
                return dict(
                    ticker=tk, price=price, ret1y=ret1y, ret1m=ret1m, slope=slope,
                    pe=pe, roe=roe, new_growth=new_growth, hist=hist, info=info
                )
            except:
                return None


        with st.spinner("Fetching stock data..."):
            data = [fetch(tk) for tk in tickers]
            data = [d for d in data if d and pd.notna(d["ret1m"]) and d["ret1m"] > 0]

        if not data:
            st.warning("No qualifying stocks found.")
            st.stop()

        df = pd.DataFrame(data)

        # 5. Scoring Model (Modify weights if needed)
        df["score"] = (
                0.3 * df["ret1y"].clip(0) +
                0.3 * (df["roe"] / 100).clip(0) +
                0.15 * (1 / df["pe"].replace(0, np.nan)) +
                0.15 * (df["slope"] / df["slope"].abs().max()) +
                0.1 * df["new_growth"] +
                0.1 * df["ret1m"]
        )

        top5 = df.sort_values("score", ascending=False).head(5)

        # 6. Output - Table
        st.subheader("📊 Top 5 Growth Stocks")
        st.table(top5[["ticker", "price", "ret1y", "ret1m", "roe", "pe", "new_growth", "score"]])

        # 7. Detailed Info
        st.markdown("### 📈 Price Trend & Info")
        for row in top5.itertuples():
            st.write(f"#### {row.ticker}")
            st.line_chart(next(d["hist"] for d in data if d["ticker"] == row.ticker))
            info = next(d["info"] for d in data if d["ticker"] == row.ticker)
            st.write(f"*Name:* {info.get('longName', '')}  |  *Sector:* {info.get('sector', '')}")

        # 8. 📌 Investment Tips
        st.markdown("---")
        st.markdown("## 📌 Final Investment Tips")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # ---- Plot 1: Risk vs Return (1Y vs 1M returns) ----
        st.markdown("### 🔍 Risk vs Return (Momentum)")

        fig, ax = plt.subplots()
        sns.scatterplot(data=top5, x="ret1y", y="ret1m", size="roe", hue="ticker", palette="Set2", ax=ax,
                        sizes=(50, 500), legend=False)
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlabel("1-Year Return")
        ax.set_ylabel("1-Month Return")
        ax.set_title("Momentum vs Long-Term Return\n(Bubble Size = ROE)")
        st.pyplot(fig)

        st.info(
            "✅ *Top-right quadrant* = High short-term + long-term returns. \n\n🟢 *Bigger bubbles = higher ROE* → better profitability.")

        # ---- Plot 2: ROE vs PE (Profitability vs Valuation) ----
        st.markdown("### 💡 Value vs Profitability (ROE vs PE)")

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=top5, x="pe", y="roe", hue="ticker", palette="Dark2", s=120, ax=ax2)
        ax2.axhline(15, color='gray', linestyle='--')
        ax2.axvline(20, color='gray', linestyle='--')
        ax2.set_xlabel("P/E Ratio (Valuation)")
        ax2.set_ylabel("ROE (%)")
        ax2.set_title("Valuation vs Profitability")
        st.pyplot(fig2)

        st.info(
            "✅ *Top-left quadrant* = High ROE + Low PE → undervalued and profitable.\n\n❌ *Bottom-right* = Overvalued & underperforming → Avoid.")

        # ---- Plot 3: Slope vs 1M Return (Trend Strength) ----
        st.markdown("### 📊 Trend Strength vs Monthly Return")

        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=top5, x="slope", y="ret1m", hue="ticker", palette="cool", s=120, ax=ax3)
        ax3.axhline(0, color='gray', linestyle='--')
        ax3.axvline(0, color='gray', linestyle='--')
        ax3.set_xlabel("Price Trend Slope (1Y)")
        ax3.set_ylabel("1-Month Return")
        ax3.set_title("Momentum Trend Strength vs Monthly Return")
        st.pyplot(fig3)

        st.info(
            "✅ *Top-right quadrant* = Positive trend + Positive return → bullish.\n\n⚠ Negative slope = caution unless backed by fundamentals.")

        # ---- Final Takeaways ----
        st.markdown("## 💼 Final Pro Investment Tips")

        st.markdown("""
        #### 📌 Strategic Insights:
        - *🎯 High ROE + Low PE = Strong value stocks* – target for long-term holding.
        - *🚀 Strong slope + High 1M Return = Momentum stocks* – suitable for swing trading.
        - *🛑 High PE & Low ROE = Overvalued with weak fundamentals* – avoid or short.
        - *📈 Price near 52-week high + rising trend = breakout watch* – confirm with volume.
        - *📉 Negative trend + poor return = declining asset* – avoid unless reversal signals emerge.

        > 📌 Use quadrant analysis + scoring metrics together for a 360° evaluation.
        > Diversify between value & growth opportunities based on market conditions.
        """)

        st.success(
            "💡 *Pro Tip:* Combine technical trends + valuation + profitability to filter the best stocks before diving deeper into financials.")
    else:
        st.write("Nothing ")

            # ===============================