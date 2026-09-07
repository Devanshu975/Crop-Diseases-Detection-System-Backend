import os
import tempfile
import uuid
import requests

from datetime import datetime, timezone, timedelta
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic import BaseModel
from ultralytics import YOLO

from risk_model import predict_risk, disease_advice


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


# ============================================================
# ENVIRONMENT VARIABLE VALIDATION
# ============================================================

if not OPENWEATHER_API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY is missing from .env"
    )

if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL is missing from .env"
    )

if not SUPABASE_SERVICE_KEY:
    raise ValueError(
        "SUPABASE_SERVICE_KEY is missing from .env"
    )


# ============================================================
# SUPABASE CLIENT
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY
)


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found at: {MODEL_PATH}"
    )

disease_model = YOLO(MODEL_PATH)


# ============================================================
# DISEASE CLASS MAPPING
# ============================================================

DISEASE_CLASSES = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Late Blight",
    3: "Leaf Mold",
    4: "Septoria Leaf Spot",
    5: "Spider Mites",
    6: "Target Spot",
    7: "Tomato Yellow Leaf Curl Virus",
    8: "Tomato Mosaic Virus",
    9: "Healthy"
}


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp"
}

STORAGE_BUCKET = "crop-images"

OPENWEATHER_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)
OPENMETEO_GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

OPENMETEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TriX CropShield Backend API",
    description=(
        "Backend API for TriX CropShield. "
        "Provides crop disease detection, "
        "live weather, environmental risk analysis "
        "and Supabase storage."
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPER: DISEASE FROM YOLO RESULT
# ============================================================

def get_disease_from_result(result):
    """
    Get the highest-confidence disease prediction
    from the YOLO result.
    """

    if result.boxes is None:
        return None, 0.0

    if len(result.boxes) == 0:
        return None, 0.0

    confidence_values = result.boxes.conf

    best_index = confidence_values.argmax().item()

    class_id = int(
        result.boxes.cls[best_index].item()
    )

    confidence = float(
        result.boxes.conf[best_index].item()
    )

    disease = DISEASE_CLASSES.get(
        class_id,
        f"Unknown Class {class_id}"
    )

    return disease, confidence


# ============================================================
# HELPER: DISEASE ADVICE
# ============================================================

def get_disease_advice(disease):
    """
    Return farmer-friendly advice based on disease.
    """

    default_advice = {
        "action": (
            "Monitor the crop regularly and "
            "inspect affected leaves."
        ),
        "warning": (
            "Follow local agricultural guidance "
            "if symptoms increase."
        )
    }

    if not disease:
        return default_advice

    disease_key = (
        disease
        .lower()
        .replace(" ", "_")
    )

    return disease_advice.get(
        disease_key,
        default_advice
    )


# ============================================================
# HELPER: OVERALL CROP STATUS
# ============================================================

def calculate_overall_status(
    disease,
    confidence,
    risk_level
):
    """
    Combine disease prediction and environmental
    risk into one farmer-friendly status.
    """

    disease = disease or "Unknown"

    # --------------------------------------------------------
    # HEALTHY CROP
    # --------------------------------------------------------

    if disease.lower() == "healthy":

        if risk_level == "High Risk":
            return (
                "Healthy currently, but environmental "
                "risk is high"
            )

        if risk_level == "Medium Risk":
            return (
                "Healthy currently, but monitoring "
                "is recommended"
            )

        return (
            "Healthy crop with favorable "
            "environmental conditions"
        )

    # --------------------------------------------------------
    # DISEASE + HIGH RISK
    # --------------------------------------------------------

    if (
        confidence >= 0.80
        and risk_level == "High Risk"
    ):
        return (
            "CRITICAL: Disease detected with "
            "high environmental risk"
        )

    # --------------------------------------------------------
    # HIGH CONFIDENCE DISEASE
    # --------------------------------------------------------

    if confidence >= 0.80:
        return (
            "Disease detected with high confidence"
        )

    # --------------------------------------------------------
    # HIGH ENVIRONMENTAL RISK
    # --------------------------------------------------------

    if risk_level == "High Risk":
        return (
            "Environmental risk is high; "
            "disease detection confidence is moderate"
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "Disease risk detected; "
        "regular monitoring recommended"
    )


# ============================================================
# HELPER: UPLOAD IMAGE TO SUPABASE
# ============================================================

def upload_image_to_supabase(
    filename,
    image_bytes,
    content_type
):
    """
    Upload crop image to Supabase Storage.
    """

    extension = os.path.splitext(
        filename
    )[1].lower()

    if not extension:
        extension = ".jpg"

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    storage_path = unique_filename

    supabase.storage \
        .from_(STORAGE_BUCKET) \
        .upload(
            storage_path,
            image_bytes,
            {
                "content-type": (
                    content_type or "image/jpeg"
                ),
                "upsert": "true"
            }
        )

    image_url = (
        supabase
        .storage
        .from_(STORAGE_BUCKET)
        .get_public_url(storage_path)
    )

    return image_url


# ============================================================
# HELPER: GET CURRENT WEATHER
# ============================================================

def get_live_weather(city):
    """
    Get current weather directly from OpenWeather.
    """

    city = city.strip()

    if not city:
        raise ValueError(
            "City name is required."
        )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        OPENWEATHER_URL,
        params=params,
        timeout=10
    )

    if response.status_code != 200:
        try:
            error_data = response.json()
        except Exception:
            error_data = {}

        message = error_data.get(
            "message",
            "Unable to fetch weather data."
        )

        raise ValueError(message)

    data = response.json()

    return {
        "city": data["name"],
        "country": data.get("sys", {}).get(
            "country",
            ""
        ),
        "temperature": float(
            data["main"]["temp"]
        ),
        "feels_like": float(
            data["main"].get(
                "feels_like",
                data["main"]["temp"]
            )
        ),
        "humidity": float(
            data["main"]["humidity"]
        ),
        "condition": data["weather"][0][
            "description"
        ],
        "weather_code": data["weather"][0][
            "id"
        ],
        "wind_kmh": round(
            float(
                data["wind"]["speed"]
            ) * 3.6,
            1
        ),
        "latitude": data["coord"]["lat"],
        "longitude": data["coord"]["lon"]
    }


# ============================================================
# HELPER: GET FORECAST
# ============================================================

def get_weather_forecast(city):
    """
    Get a 7-day daily weather forecast using Open-Meteo.

    Open-Meteo provides 7 forecast days by default.
    We first convert the city name into coordinates
    using the Open-Meteo geocoding API.
    """

    city = city.strip()

    if not city:
        raise ValueError("City name is required.")

    # --------------------------------------------------------
    # 1. FIND CITY COORDINATES
    # --------------------------------------------------------

    geocoding_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geocoding_response = requests.get(
        OPENMETEO_GEOCODING_URL,
        params=geocoding_params,
        timeout=10
    )

    if geocoding_response.status_code != 200:
        raise ValueError(
            "Unable to find city location."
        )

    geocoding_data = geocoding_response.json()

    locations = geocoding_data.get(
        "results",
        []
    )

    if not locations:
        raise ValueError(
            f"City '{city}' could not be found."
        )

    location = locations[0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    # --------------------------------------------------------
    # 2. GET 7-DAY FORECAST
    # --------------------------------------------------------

    forecast_params = {
        "latitude": latitude,
        "longitude": longitude,

        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "weather_code,"
            "rain_sum"
        ),

        "timezone": "auto",

        "forecast_days": 7
    }

    forecast_response = requests.get(
        OPENMETEO_FORECAST_URL,
        params=forecast_params,
        timeout=10
    )

    if forecast_response.status_code != 200:
        try:
            error_data = forecast_response.json()
        except Exception:
            error_data = {}

        raise ValueError(
            error_data.get(
                "reason",
                "Unable to fetch 7-day forecast."
            )
        )

    forecast_data = forecast_response.json()

    daily = forecast_data.get(
        "daily",
        {}
    )

    dates = daily.get(
        "time",
        []
    )

    highs = daily.get(
        "temperature_2m_max",
        []
    )

    lows = daily.get(
        "temperature_2m_min",
        []
    )

    weather_codes = daily.get(
        "weather_code",
        []
    )

    rainfall = daily.get(
        "rain_sum",
        []
    )

    # --------------------------------------------------------
    # 3. VALIDATE DATA
    # --------------------------------------------------------

    if not dates:
        raise ValueError(
            "No forecast data received."
        )

    # --------------------------------------------------------
    # 4. WEATHER CODE → DESCRIPTION
    # --------------------------------------------------------

    def weather_description(code):
        weather_codes_map = {

            0: "clear sky",

            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",

            45: "fog",
            48: "depositing rime fog",

            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",

            56: "light freezing drizzle",
            57: "dense freezing drizzle",

            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",

            66: "light freezing rain",
            67: "heavy freezing rain",

            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",

            77: "snow grains",

            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",

            85: "slight snow showers",
            86: "heavy snow showers",

            95: "thunderstorm",

            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail"
        }

        return weather_codes_map.get(
            int(code),
            "unknown"
        )

    # --------------------------------------------------------
    # 5. BUILD FRONTEND-FRIENDLY RESPONSE
    # --------------------------------------------------------

    result = []

    for index in range(
        min(7, len(dates))
    ):

        code = int(
            weather_codes[index]
        )

        rain = (
            rainfall[index]
            if index < len(rainfall)
            else 0
        )

        result.append({

            "date": dates[index],

            "high": round(
                float(highs[index]),
                1
            ),

            "low": round(
                float(lows[index]),
                1
            ),

            "rainfall_mm": round(
                float(rain or 0),
                1
            ),

            "weather_code": code,

            "condition": weather_description(
                code
            )
        })

    return result


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "success",
        "message": "TriX CropShield Backend is running",
        "version": "2.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "TriX backend is working"
    }


# ============================================================
# TEST SUPABASE
# ============================================================

@app.get("/test-supabase")
def test_supabase():

    try:

        response = (
            supabase
            .table("crop_scans")
            .select("*")
            .limit(1)
            .execute()
        )

        return {
            "status": "success",
            "message": (
                "Supabase connection is working"
            ),
            "data": response.data
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# WEATHER
# ============================================================

@app.get("/weather")
def weather_endpoint(
    city: str = Query(...)
):

    try:

        current = get_live_weather(city)

        forecast = get_weather_forecast(city)

        return {
            "status": "success",

            "data": {

                "city": current["city"],

                "country": current["country"],

                "temperature": current[
                    "temperature"
                ],

                "feels_like": current[
                    "feels_like"
                ],

                "humidity": current[
                    "humidity"
                ],

                "condition": current[
                    "condition"
                ],

                "weather_code": current[
                    "weather_code"
                ],

                "wind_kmh": current[
                    "wind_kmh"
                ],

                "latitude": current[
                    "latitude"
                ],

                "longitude": current[
                    "longitude"
                ],

                "forecast": forecast
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CROP SCAN MODEL
# ============================================================

class CropScan(BaseModel):

    crop: str
    image_url: str
    disease: str
    confidence: float


# ============================================================
# CREATE CROP SCAN
# ============================================================

@app.post("/crop-scan")
def create_crop_scan(
    scan: CropScan
):

    try:

        response = (
            supabase
            .table("crop_scans")
            .insert({
                "crop": scan.crop,
                "image_url": scan.image_url,
                "disease": scan.disease,
                "confidence": float(
                    scan.confidence
                )
            })
            .execute()
        )

        return {
            "status": "success",
            "message": "Crop scan saved",
            "data": response.data
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# RISK REQUEST MODEL
# ============================================================

class RiskRequest(BaseModel):

    scan_id: int
    city: str
    crop: str


# ============================================================
# PREDICT CROP RISK
# ============================================================

@app.post("/predict-risk")
def predict_crop_risk(
    request: RiskRequest
):

    try:

        result = predict_risk(
            request.city,
            request.crop
        )

        risk_level = str(
            result.get(
                "risk_level",
                "Low Risk"
            )
        )

        risk_score = float(
            result.get(
                "risk_score",
                0
            )
        )

        temperature = float(
            result.get(
                "temperature",
                0
            )
        )

        humidity = float(
            result.get(
                "humidity",
                0
            )
        )

        recommendation = str(
            result.get(
                "recommendation",
                "Monitor the crop regularly."
            )
        )

        response = (
            supabase
            .table("risk_predictions")
            .insert({
                "scan_id": int(
                    request.scan_id
                ),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "reason": (
                    f"Temperature: "
                    f"{temperature}°C, "
                    f"Humidity: "
                    f"{humidity}%"
                ),
                "recommendation": recommendation
            })
            .execute()
        )

        return {

            "status": "success",

            "message": (
                "Risk prediction saved"
            ),

            "data": {

                "risk_prediction": {

                    "risk_level": risk_level,

                    "risk_score": risk_score,

                    "temperature": temperature,

                    "humidity": humidity,

                    "recommendation": (
                        recommendation
                    )
                },

                "database_record": (
                    response.data
                )
            }
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# PREDICT DISEASE
# ============================================================

@app.post("/predict-disease")
async def predict_disease(
    crop: str,
    file: UploadFile = File(...)
):

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a JPG, JPEG, PNG "
                "or WEBP image."
            )
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # ----------------------------------------------------
        # TEMPORARY FILE
        # ----------------------------------------------------

        extension = os.path.splitext(
            file.filename or ".jpg"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                image_bytes
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = disease_model.predict(
            source=temp_path,
            verbose=False
        )

        result = results[0]

        disease, confidence = (
            get_disease_from_result(
                result
            )
        )

        # ----------------------------------------------------
        # NO DETECTION
        # ----------------------------------------------------

        if disease is None:

            return {

                "status": "success",

                "message": (
                    "No disease detected"
                ),

                "data": {

                    "crop": crop,

                    "disease": "No detection",

                    "confidence": 0.0,

                    "confidence_percent": 0.0
                }
            }

        # ----------------------------------------------------
        # UPLOAD IMAGE
        # ----------------------------------------------------

        image_url = (
            upload_image_to_supabase(
                file.filename or "crop_image.jpg",
                image_bytes,
                file.content_type
            )
        )

        # ----------------------------------------------------
        # SAVE DATABASE
        # ----------------------------------------------------

        database_response = (
            supabase
            .table("crop_scans")
            .insert({
                "crop": crop,
                "image_url": image_url,
                "disease": disease,
                "confidence": float(
                    confidence
                )
            })
            .execute()
        )

        scan_id = None

        if database_response.data:

            scan_id = (
                database_response
                .data[0]
                .get("id")
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "message": (
                "Disease prediction completed "
                "and saved"
            ),

            "data": {

                "scan_id": scan_id,

                "crop": crop,

                "disease": disease,

                "confidence": float(
                    confidence
                ),

                "confidence_percent": round(
                    float(confidence) * 100,
                    2
                ),

                "image_url": image_url,

                "database_record": (
                    database_response.data
                )
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# COMPLETE CROP HEALTH ANALYSIS
# ============================================================

@app.post("/complete-analysis")
async def complete_analysis(

    file: UploadFile = File(...),

    crop: str = Query(...),

    city: str = Query(...)

):

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a JPG, JPEG, PNG "
                "or WEBP image."
            )
        )

    temp_path = None

    try:

        # ====================================================
        # 1. READ IMAGE
        # ====================================================

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # ====================================================
        # 2. TEMPORARY FILE
        # ====================================================

        extension = os.path.splitext(
            file.filename or ".jpg"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                image_bytes
            )

            temp_path = temp_file.name

        # ====================================================
        # 3. YOLO DISEASE PREDICTION
        # ====================================================

        results = disease_model.predict(
            source=temp_path,
            verbose=False
        )

        result = results[0]

        disease, confidence = (
            get_disease_from_result(
                result
            )
        )

        # ====================================================
        # 4. NO DETECTION
        # ====================================================

        if disease is None:

            return {

                "status": "success",

                "message": (
                    "No disease could be detected "
                    "in the uploaded image."
                ),

                "data": {

                    "crop": crop,

                    "city": city,

                    "disease": "No detection",

                    "confidence": 0.0,

                    "confidence_percent": 0.0,

                    "overall_status": (
                        "Unable to determine crop health"
                    )
                }
            }

        # ====================================================
        # 5. UPLOAD IMAGE
        # ====================================================

        image_url = (
            upload_image_to_supabase(
                file.filename or "crop_image.jpg",
                image_bytes,
                file.content_type
            )
        )

        # ====================================================
        # 6. SAVE CROP SCAN
        # ====================================================

        scan_response = (
            supabase
            .table("crop_scans")
            .insert({
                "crop": crop,
                "image_url": image_url,
                "disease": disease,
                "confidence": float(
                    confidence
                )
            })
            .execute()
        )

        if not scan_response.data:

            return {

                "status": "error",

                "message": (
                    "Crop scan could not be saved."
                )
            }

        scan_id = (
            scan_response
            .data[0]["id"]
        )

        # ====================================================
        # 7. ENVIRONMENTAL RISK
        # ====================================================

        risk_result = predict_risk(
            city,
            crop
        )

        risk_level = str(
            risk_result.get(
                "risk_level",
                "Low Risk"
            )
        )

        risk_score = float(
            risk_result.get(
                "risk_score",
                0
            )
        )

        temperature = float(
            risk_result.get(
                "temperature",
                0
            )
        )

        humidity = float(
            risk_result.get(
                "humidity",
                0
            )
        )

        recommendation = str(
            risk_result.get(
                "recommendation",
                "Monitor the crop regularly."
            )
        )

        # ====================================================
        # 8. DISEASE ADVICE
        # ====================================================

        advice = get_disease_advice(
            disease
        )

        # ====================================================
        # 9. OVERALL STATUS
        # ====================================================

        overall_status = (
            calculate_overall_status(
                disease,
                float(confidence),
                risk_level
            )
        )

        # ====================================================
        # 10. SAVE RISK PREDICTION
        # ====================================================

        risk_response = (
            supabase
            .table("risk_predictions")
            .insert({

                "scan_id": int(
                    scan_id
                ),

                "risk_level": risk_level,

                "risk_score": risk_score,

                "reason": (
                    f"Disease: {disease}, "
                    f"Confidence: "
                    f"{round(float(confidence) * 100, 2)}%, "
                    f"Temperature: "
                    f"{temperature}°C, "
                    f"Humidity: "
                    f"{humidity}%"
                ),

                "recommendation": (
                    recommendation
                )

            })
            .execute()
        )

        # ====================================================
        # 11. FINAL RESPONSE
        # ====================================================

        return {

            "status": "success",

            "message": (
                "Complete crop analysis successful"
            ),

            "data": {

                "scan_id": scan_id,

                "crop": crop,

                "city": risk_result.get(
                    "city",
                    city
                ),

                "disease": disease,

                "confidence": float(
                    confidence
                ),

                "confidence_percent": round(
                    float(confidence) * 100,
                    2
                ),

                "image_url": image_url,

                "temperature": temperature,

                "humidity": humidity,

                "risk_level": risk_level,

                "risk_score": risk_score,

                "overall_status": (
                    overall_status
                ),

                "recommendation": (
                    recommendation
                ),

                "action": advice[
                    "action"
                ],

                "warning": advice[
                    "warning"
                ],

                "risk_database_record": (
                    risk_response.data
                )
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# GET SCAN HISTORY
# ============================================================

@app.get("/scan-history")
def get_scan_history():

    try:

        response = (
            supabase
            .table("crop_scans")
            .select("*")
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        return {

            "status": "success",

            "data": response.data
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

@app.get("/analysis-summary")
def analysis_summary(
    scan_id: int = Query(...)
):

    try:

        # ----------------------------------------------------
        # GET CROP SCAN
        # ----------------------------------------------------

        scan_response = (
            supabase
            .table("crop_scans")
            .select("*")
            .eq(
                "id",
                int(scan_id)
            )
            .limit(1)
            .execute()
        )

        if not scan_response.data:

            return {

                "status": "error",

                "message": "Scan not found"
            }

        scan = scan_response.data[0]

        # ----------------------------------------------------
        # GET RISK
        # ----------------------------------------------------

        risk_response = (
            supabase
            .table("risk_predictions")
            .select("*")
            .eq(
                "scan_id",
                int(scan_id)
            )
            .order(
                "created_at",
                desc=True
            )
            .limit(1)
            .execute()
        )

        if not risk_response.data:

            return {

                "status": "error",

                "message": (
                    "Risk prediction not found"
                )
            }

        risk = risk_response.data[0]

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        risk_level = str(
            risk.get(
                "risk_level",
                "Low Risk"
            )
        )

        if risk_level == "High Risk":

            status = "CRITICAL"

        elif risk_level == "Medium Risk":

            status = "WARNING"

        else:

            status = "SAFE"

        # ----------------------------------------------------
        # DISEASE ADVICE
        # ----------------------------------------------------

        disease = scan.get(
            "disease",
            "Unknown"
        )

        advice = get_disease_advice(
            disease
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            scan.get(
                "confidence",
                0
            )
        )

        risk_score = float(
            risk.get(
                "risk_score",
                0
            )
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "data": {

                "scan_id": scan["id"],

                "crop": scan["crop"],

                "disease": scan["disease"],

                "confidence_percent": round(
                    confidence * 100,
                    2
                ),

                "risk_level": risk_level,

                "risk_score": risk_score,

                "overall_status": status,

                "recommendation": risk.get(
                    "recommendation",
                    "Monitor the crop regularly."
                ),

                "action": advice[
                    "action"
                ],

                "warning": advice[
                    "warning"
                ],

                "image_url": scan[
                    "image_url"
                ]
            }
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )