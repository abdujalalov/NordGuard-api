import os

import inflection
import joblib
from dotenv import load_dotenv

load_dotenv()


class FraudDetection:
    def __init__(self):
        self.scaler = joblib.load(os.getenv("SCALER_PATH"))
        self.encoder = joblib.load(os.getenv("ENCODER_PATH"))

    @staticmethod
    def get_model():
        return joblib.load(os.getenv("ML_MODEL_PATH"))

    def clean_data(self, df):
        df.columns = [inflection.underscore(col) for col in df.columns]
        df['diff_new_old_balance'] = df['newbalance_orig'] - df['oldbalance_org']
        df['diff_new_old_destiny'] = df['newbalance_dest'] - df['oldbalance_dest']
        return df

    def prepare_data(self, df):
        df_cleaned = self.clean_data(df)

        X = df_cleaned.drop(columns=['is_fraud', 'is_flagged_fraud', 'name_orig', 'name_dest'], axis=1, errors='ignore')

        # One Hot Encoding
        X_encoded = self.encoder.transform(X)

        # Rescaling
        num_columns = ['amount', 'oldbalance_org', 'newbalance_orig', 'oldbalance_dest', 'newbalance_dest',
                       'diff_new_old_balance', 'diff_new_old_destiny']

        X_encoded[num_columns] = self.scaler.transform(X_encoded[num_columns])

        # Feature Selection
        selected_columns = ['step', 'oldbalance_org',
                            'newbalance_orig', 'newbalance_dest',
                            'diff_new_old_balance', 'diff_new_old_destiny',
                            'type_TRANSFER']

        X_selected = X_encoded[selected_columns]

        return X_selected

    def predict(self, model, original_data, test_data):
        predictions = model.predict(test_data)
        original_data['prediction'] = predictions
        return original_data.to_json(orient="records", date_format="iso")
