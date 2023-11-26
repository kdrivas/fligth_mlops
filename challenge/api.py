from fastapi import Request, FastAPI, HTTPException
import pandas as pd

from .model import DelayModel
from .constants import OHE_VALUES, FEATURES_COLS
from .data_validation import get_invalid_columns

app = FastAPI()

model = DelayModel()
model.init_model("artifacts/model.pkl")


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
            detail=f"The following columns are missing: {list(df_data.columns) - OHE_VALUES.keys()}",
        )

    # Validate if the values are the expected ones
    if len(get_invalid_columns(df_data)):
        raise HTTPException(
            status_code=400,
            detail=f"Some of the columns don't contain the expected categories: {get_invalid_columns}",
        )

    df_features = model.preprocess(df_data)
    prediction = model.predict(df_features).tolist()

    return {"predict": prediction}
