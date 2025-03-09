import unittest
import joblib
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class TestTimeSeriesModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load or generate synthetic time series data"""
        np.random.seed(42)
        time_series_data = np.cumsum(np.random.randn(1000))  # Simulated time series
        cls.scaler = MinMaxScaler()
        cls.data_scaled = cls.scaler.fit_transform(time_series_data.reshape(-1, 1))
        
        cls.look_back = 20
        cls.X, cls.y = cls.create_features(cls.data_scaled, cls.look_back)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    @staticmethod
    def create_features(data, look_back):
        """Generate time series features for LSTM & XGBoost"""
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def test_sarima_model(self):
        """Train SARIMA model and check its ability to forecast"""
        sarima_model = SARIMAX(self.y_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        sarima_result = sarima_model.fit(disp=False)

        # Forecast next values
        sarima_forecast = sarima_result.forecast(steps=len(self.y_test))
        mae = mean_absolute_error(self.y_test, sarima_forecast)

        joblib.dump(sarima_result, "sarima_model.pkl")
        self.assertLess(mae, 0.5, "SARIMA Mean Absolute Error should be < 0.5")

    def test_xgboost_model(self):
        """Train XGBoost for time series forecasting"""
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        xgb_model.fit(self.X_train.reshape(self.X_train.shape[0], -1), self.y_train)

        y_pred_xgb = xgb_model.predict(self.X_test.reshape(self.X_test.shape[0], -1))
        y_pred_xgb = self.scaler.inverse_transform(y_pred_xgb.reshape(-1, 1))

        mae = mean_absolute_error(self.y_test, y_pred_xgb)
        self.assertLess(mae, 0.5, "XGBoost MAE should be < 0.5")

    def test_lstm_model(self):
        """Train LSTM model for time series forecasting"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(self.look_back, 1)),
            tf.keras.layers.LSTM(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=0)

        y_pred_lstm = model.predict(self.X_test)
        y_pred_lstm = self.scaler.inverse_transform(y_pred_lstm)

        mae = mean_absolute_error(self.y_test, y_pred_lstm)
        self.assertLess(mae, 0.5, "LSTM MAE should be < 0.5")


if __name__ == "__main__":
    unittest.main()

