import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class FraudAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def generate_sample_data(self, n_transactions=10000):
        # Generate dates
        dates = [datetime.now() - timedelta(days=x) for x in range(30)]
        
        # Create transactions
        data = {
            'transaction_id': range(n_transactions),
            'date': np.random.choice(dates, n_transactions),
            'amount': np.random.exponential(1000, n_transactions),
            'latitude': np.random.uniform(-23.6, -22.5, n_transactions),
            'longitude': np.random.uniform(-46.8, -45.7, n_transactions)
        }
        
        # Add fraud labels (5% fraud rate)
        data['is_fraud'] = np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
        
        # Increase amounts for fraudulent transactions
        fraud_idx = data['is_fraud'] == 1
        data['amount'][fraud_idx] *= 2.5
        
        return pd.DataFrame(data)

    def train_model(self, df):
        X = df[['amount', 'latitude', 'longitude']]
        y = df['is_fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        
        # Generate confusion matrix
        y_pred = self.model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            text=conf_matrix,
            texttemplate="%{text}",
            colorscale='Reds'
        ))
        fig.update_layout(title='Confusion Matrix')
        fig.write_html('confusion_matrix.html')
        
        return X_test, y_test

    def analyze_transactions(self, df):
        # Get fraud probabilities
        probs = self.model.predict_proba(df[['amount', 'latitude', 'longitude']])[:, 1]
        df['risk_score'] = probs
        
        # Geographic distribution
        fig = px.scatter_geo(df,
                              lat='latitude',
                              lon='longitude',
                              color='risk_score',
                              size='amount',
                              hover_data=['transaction_id', 'amount'],
                              title='Geographic Distribution of Transactions',
                              projection='mercator',
                              color_continuous_scale='Viridis')
        fig.update_geos(
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="LightGray",
            showocean=True,
            oceancolor="LightBlue"
        )

        fig.update_layout(
            geo=dict(
                center=dict(lat=df['latitude'].mean(), 
                           lon=df['longitude'].mean()),
                projection_scale=50
            )
        )
        fig.write_html('geographic_distribution.html')
        
        # High-risk transactions
        high_risk = df[df['risk_score'] > 0.7].sort_values('risk_score', ascending=False)
        fig = px.scatter(high_risk,
                        x='amount',
                        y='risk_score',
                        title='High-Risk Transactions',
                        color='risk_score',
                        size='amount')
        fig.write_html('high_risk_transactions.html')
        
        return high_risk

# Run the analysis
analyzer = FraudAnalyzer()
data = analyzer.generate_sample_data()
X_test, y_test = analyzer.train_model(data)
high_risk_transactions = analyzer.analyze_transactions(data)

print("Analysis complete! Check the generated HTML files for visualizations.")
