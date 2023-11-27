from fastapi import Request, FastAPI, HTTPException
import uvicorn
import pandas as pd
import os

from challenge.constants import OHE_VALUES, FEATURES_COLS
from challenge.data_validation import get_invalid_columns
from challenge.model import DelayModel

app = FastAPI()

model = DelayModel()
model_path = os.environ.get("MODEL_PATH", "artifacts/model.pkl")
model.init_model(model_path)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    data = await request.json()
    df_data = pd.DataFrame(data["flights"])

    # Check all the raw columns are in the df
    if list(df_data.columns) != list(OHE_VALUES.keys()):
        raise HTTPException(
            status_code=400,
            detail="The following columns are missing:"
            f" {list(df_data.columns) - OHE_VALUES.keys()}",
        )

    # Validate if the values are the expected ones
    if len(get_invalid_columns(df_data)):
        raise HTTPException(
            status_code=400,
            detail="Some of the columns don't contain the expected"
            f" categories: {get_invalid_columns}",
        )

    df_features = model.preprocess(df_data)
    prediction = model.predict(df_features).tolist()

    return {"predict": prediction}
