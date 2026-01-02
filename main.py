#--------------------------------imports--------------------------------#

import yfinance as yf
import pandas as pd 
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt 

#------------------------------lets get the data-------------------------------#
def get_data():
    data = yf.download(tickers = 'BTC-INR', period = '1d', interval = '1m')
    return data[['Close', 'Volume']]

#------------------------------anomaly detection-------------------------------

def detect_anomalies(df):
    model = IsolationForest(contamination=0.05 , random_state=42)
    df['anomaly_signal'] = model.fit_predict(df[['Close', 'Volume']])    #unsupervised training
    df['scores'] = model.decision_function(df[['Close', 'Volume']])      #anomaly scores (low the score more the weird)
    return df

#-----------------------bhagaaaaaaoooooo-----------------------------#

df = get_data()
df_analyzed = detect_anomalies(df)

#------------------------------visualization-------------------------------#

#subplot 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10), sharex=True)

# TOP PLOT: Price
ax1.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.6)
anomalies = df[df['anomaly_signal'] == -1]
ax1.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomaly', zorder=5)
ax1.set_ylabel("Price (USD)")
ax1.set_title("Bitcoin Real-time Analysis (Price & Volume Anomalies)")
ax1.legend()

# BOTTOM PLOT: Volume (as a bar chart)
ax2.bar(df.index, df['Volume'], color='gray', alpha=0.3, label='Volume')
# Highlight volume spikes that the model flagged
ax2.scatter(anomalies.index, anomalies['Volume'], color='red', s=10)
ax2.set_ylabel("Volume")
ax2.set_xlabel("Time (UTC)")

plt.tight_layout()
plt.show()



