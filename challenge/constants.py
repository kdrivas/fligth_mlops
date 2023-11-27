# Data validation
OPERA_VALID_VALUES = [
    "American Airlines",
    "Air Canada",
    "Air France",
    "Aeromexico",
    "Aerolineas Argentinas",
    "Austral",
    "Avianca",
    "Alitalia",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Iberia",
    "K.L.M.",
    "Qantas Airways",
    "United Airlines",
    "Grupo LATAM",
    "Sky Airline",
    "Latin American Wings",
    "Plus Ultra Lineas Aereas",
    "JetSmart SPA",
    "Oceanair Linhas Aereas",
    "Lacsa",
]
TIPOVUELO_VALID_VALUES = ["I", "N"]
MES_VALID_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

VALID_VALUES = {
    "OPERA": OPERA_VALID_VALUES,
    "TIPOVUELO": TIPOVUELO_VALID_VALUES,
    "MES": MES_VALID_VALUES,
}


# Preprocessing
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
THRESHOLD_MINUTES = 15

OHE_VALUES = {
    "OPERA": ["Latin American Wings", "Sky Airline", "Copa Air", "Grupo LATAM"],
    "TIPOVUELO": ["I"],
    "MES": [4, 7, 10, 11, 12],
}

# Training and prediction
FEATURES_COLS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]
