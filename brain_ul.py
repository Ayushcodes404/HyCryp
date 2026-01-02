import pandas as pd
from sklearn.ensemble import IsolationForest

class MarketRadar:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def extract_signals(self, df):
        """
        Takes raw Price/Volume data and returns 'Weirdness' scores.
        This is the core LINK between the two systems.
        """
        # Ensure we only use numeric columns for the UL model
        data_to_fit = df[['Close', 'Volume']]
        
        # Unsupervised fit and predict
        df['anomaly_signal'] = self.model.fit_predict(data_to_fit)
        df['anomaly_score'] = self.model.decision_function(data_to_fit)
        
        return df[['Close', 'Volume', 'anomaly_signal', 'anomaly_score']]