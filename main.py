import os
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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


# ============================================================
# SUPABASE CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is missing from .env")

if not SUPABASE_KEY:
    raise ValueError("SUPABASE_SERVICE_KEY is missing from .env")


# ============================================================
# CREATE SUPABASE CLIENT
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
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


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TriX Crop Disease Detection API",
    description=(
        "Backend API for TriX CropShield "
        "crop disease and environmental risk analysis."
    ),
    version="1.0.0"
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
# HELPER: GET DISEASE FROM YOLO RESULT
# ============================================================

def get_disease_from_result(result):
    """
    Get the highest-confidence detection from YOLO.
    """

    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0

    best_index = result.boxes.conf.argmax().item()

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
    Return disease-specific farmer-friendly advice.
    """

    if not disease:
        return {
            "action": (
                "Monitor the crop regularly "
                "and inspect affected leaves."
            ),
            "warning": (
                "Follow local agricultural guidance "
                "if symptoms increase."
            )
        }

    disease_key = (
        disease.lower()
        .replace(" ", "_")
    )

    return disease_advice.get(
        disease_key,
        {
            "action": (
                "Monitor the crop regularly "
                "and inspect affected leaves."
            ),
            "warning": (
                "Follow local agricultural guidance "
                "if symptoms increase."
            )
        }
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
    Combine disease confidence and environmental
    risk into a simple farmer-friendly status.
    """

    disease = disease or "Unknown"

    if disease.lower() == "healthy":

        if risk_level == "High Risk":
            return (
                "Healthy currently, but environmental "
                "risk is high"
            )

        elif risk_level == "Medium Risk":
            return (
                "Healthy currently, but monitoring "
                "is recommended"
            )

        else:
            return (
                "Healthy crop with favorable "
                "environmental conditions"
            )

    if (
        confidence >= 0.80
        and risk_level == "High Risk"
    ):
        return (
            "CRITICAL: Disease detected with "
            "high environmental risk"
        )

    elif confidence >= 0.80:
        return (
            "Disease detected with high confidence"
        )

    elif risk_level == "High Risk":
        return (
            "Environmental risk is high; "
            "disease detection confidence is moderate"
        )

    else:
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
    Upload image to Supabase Storage.

    IMPORTANT:
    No user_id is used because the crop_scans
    table does not contain a user_id column.
    """

    extension = os.path.splitext(filename)[1]

    if not extension:
        extension = ".jpg"

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    storage_path = unique_filename

    supabase.storage.from_(STORAGE_BUCKET).upload(
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
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "success",
        "message": "TriX Backend is running",
        "version": "1.0.0"
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
            "message": "Supabase connection is working",
            "data": response.data
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


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
def create_crop_scan(scan: CropScan):

    try:

        response = (
            supabase
            .table("crop_scans")
            .insert({
                "crop": scan.crop,
                "image_url": scan.image_url,
                "disease": scan.disease,
                "confidence": float(scan.confidence)
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
def predict_crop_risk(request: RiskRequest):

    try:

        result = predict_risk(
            request.city,
            request.crop
        )

        risk_level = str(
            result.get("risk_level", "Low Risk")
        )

        risk_score = float(
            result.get("risk_score", 0)
        )

        temperature = float(
            result.get("temperature", 0)
        )

        humidity = float(
            result.get("humidity", 0)
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
                "scan_id": int(request.scan_id),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "reason": (
                    f"Temperature: {temperature}°C, "
                    f"Humidity: {humidity}%"
                ),
                "recommendation": recommendation
            })
            .execute()
        )

        return {

            "status": "success",

            "message": "Risk prediction saved",

            "data": {

                "risk_prediction": {
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "temperature": temperature,
                    "humidity": humidity,
                    "recommendation": recommendation
                },

                "database_record": response.data
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
                "Please upload a JPG, JPEG, PNG, "
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
        # CREATE TEMPORARY FILE
        # ----------------------------------------------------

        extension = os.path.splitext(
            file.filename or ".jpg"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = disease_model.predict(
            source=temp_path,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # GET DISEASE
        # ----------------------------------------------------

        disease, confidence = (
            get_disease_from_result(result)
        )

        # ----------------------------------------------------
        # NO DETECTION
        # ----------------------------------------------------

        if disease is None:

            return {

                "status": "success",

                "message": "No disease detected",

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

        image_url = upload_image_to_supabase(
            file.filename or "crop_image.jpg",
            image_bytes,
            file.content_type
        )

        # ----------------------------------------------------
        # SAVE CROP SCAN
        # ----------------------------------------------------

        database_response = (
            supabase
            .table("crop_scans")
            .insert({
                "crop": crop,
                "image_url": image_url,
                "disease": disease,
                "confidence": float(confidence)
            })
            .execute()
        )

        # ----------------------------------------------------
        # GET SCAN ID
        # ----------------------------------------------------

        scan_id = None

        if database_response.data:

            scan_id = database_response.data[0].get("id")

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "status": "success",

            "message": (
                "Disease prediction completed and saved"
            ),

            "data": {

                "scan_id": scan_id,

                "crop": crop,

                "disease": disease,

                "confidence": float(confidence),

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
                "Please upload a JPG, JPEG, PNG, "
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
        # 2. CREATE TEMPORARY IMAGE
        # ====================================================

        extension = os.path.splitext(
            file.filename or ".jpg"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        # ====================================================
        # 3. DISEASE PREDICTION
        # ====================================================

        results = disease_model.predict(
            source=temp_path,
            verbose=False
        )

        result = results[0]

        disease, confidence = (
            get_disease_from_result(result)
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

        image_url = upload_image_to_supabase(

            file.filename or "crop_image.jpg",

            image_bytes,

            file.content_type
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

                "confidence": float(confidence)

            })

            .execute()
        )

        # ====================================================
        # 7. CHECK DATABASE RESULT
        # ====================================================

        if not scan_response.data:

            return {

                "status": "error",

                "message": (
                    "Crop scan could not be saved."
                )
            }

        scan_id = scan_response.data[0]["id"]

        # ====================================================
        # 8. ENVIRONMENTAL RISK
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
        # 9. DISEASE ADVICE
        # ====================================================

        advice = get_disease_advice(
            disease
        )

        # ====================================================
        # 10. OVERALL STATUS
        # ====================================================

        overall_status = calculate_overall_status(

            disease,

            float(confidence),

            risk_level
        )

        # ====================================================
        # 11. SAVE RISK PREDICTION
        # ====================================================

        risk_response = (

            supabase

            .table("risk_predictions")

            .insert({

                "scan_id": int(scan_id),

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

                "recommendation": recommendation

            })

            .execute()
        )

        # ====================================================
        # 12. FINAL RESPONSE
        # ====================================================

        return {

            "status": "success",

            "message": (
                "Complete crop analysis successful"
            ),

            "data": {

                # Identification

                "scan_id": scan_id,

                "crop": crop,

                "city": risk_result.get(
                    "city",
                    city
                ),

                # Disease

                "disease": disease,

                "confidence": float(confidence),

                "confidence_percent": round(

                    float(confidence) * 100,

                    2
                ),

                # Image

                "image_url": image_url,

                # Weather

                "temperature": temperature,

                "humidity": humidity,

                # Risk

                "risk_level": risk_level,

                "risk_score": risk_score,

                # Overall

                "overall_status": overall_status,

                # Recommendations

                "recommendation": recommendation,

                "action": advice["action"],

                "warning": advice["warning"],

                # Database

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

        # ====================================================
        # GET CROP SCAN
        # ====================================================

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

        # ====================================================
        # GET RISK PREDICTION
        # ====================================================

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

        # ====================================================
        # OVERALL STATUS
        # ====================================================

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

        # ====================================================
        # DISEASE ADVICE
        # ====================================================

        disease = scan.get(
            "disease",
            "Unknown"
        )

        advice = get_disease_advice(
            disease
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

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

        # ====================================================
        # RESPONSE
        # ====================================================

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

                "action": advice["action"],

                "warning": advice["warning"],

                "image_url": scan["image_url"]
            }
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }