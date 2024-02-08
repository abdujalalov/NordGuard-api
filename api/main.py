import os
import warnings
from typing import List, Union, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sklearn.exceptions import InconsistentVersionWarning
from dotenv import load_dotenv

from ml_utils.model_operations import FraudDetection

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_model():
    app.model = joblib.load(os.getenv("ML_MODEL_PATH"))

@app.post('/predict')
async def detect_fraud(input_json: Union[dict, List[dict]], model: Any = Depends(FraudDetection.get_model)):
    try:
        # Extract the 'data' key from the received JSON
        input_data = input_json.get('data', None)

        if input_data is None:
            raise HTTPException(status_code=422, detail="Missing 'data' key in JSON")

        if isinstance(input_data, dict): # if single row
            input_data = [input_data]
        elif not isinstance(input_data, list): # if multiple rows
            raise HTTPException(status_code=422, detail="'data' should be a list of dictionaries")

        input_data = pd.DataFrame(input_data)

        fraud_handler = FraudDetection()

        # convert numeric columns to appropriate types
        numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        input_data[numeric_columns] = input_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        prepared_data = fraud_handler.prepare_data(input_data)

        predictions = fraud_handler.predict(model, input_data, prepared_data)

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get('/')
def root():
    return {'message': 'Welcome to the Fraud Detection API v1.0.0 by NordGuard'}


# if __name__ == '__main__':
#     import uvicorn
#
#     uvicorn.run(app, host='localhost', port=8000)
