from sklearn.ensemble import RandomForestClassifier
import urllib.parse
import urllib.request
import json


# ============================================================
# 1. REAL WEATHER API
# ============================================================

def get_weather(city):
    """
    Get REAL current weather using Open-Meteo.

    Returns:
        temperature
        humidity
        weather_code
    """

    city_clean = city.strip()

    if not city_clean:
        raise ValueError("City name is required.")

    try:
        # --------------------------------------------------------
        # STEP 1: Convert city name -> latitude/longitude
        # --------------------------------------------------------

        geocode_params = urllib.parse.urlencode({
            "name": city_clean,
            "count": 1,
            "language": "en",
            "format": "json",
            "countryCode": "IN"
        })

        geocode_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + geocode_params
        )

        with urllib.request.urlopen(
            geocode_url,
            timeout=10
        ) as response:

            geocode_data = json.loads(
                response.read().decode("utf-8")
            )

        locations = geocode_data.get("results", [])

        if not locations:
            raise ValueError(
                f"Could not find weather location for '{city_clean}'."
            )

        location = locations[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # --------------------------------------------------------
        # STEP 2: Get CURRENT weather
        # --------------------------------------------------------

        weather_params = urllib.parse.urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code"
            ),
            "timezone": "auto"
        })

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + weather_params
        )

        with urllib.request.urlopen(
            weather_url,
            timeout=10
        ) as response:

            weather_data = json.loads(
                response.read().decode("utf-8")
            )

        current = weather_data.get("current", {})

        temperature = float(
            current.get("temperature_2m", 0)
        )

        humidity = float(
            current.get("relative_humidity_2m", 0)
        )

        weather_code = int(
            current.get("weather_code", 0)
        )

        return {
            "temp": temperature,
            "humidity": humidity,
            "weather_code": weather_code,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location.get(
                "name",
                city_clean
            )
        }

    except Exception as e:

        print(
            f"Weather API error for {city_clean}: {e}"
        )

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------
        # If internet/weather API fails, use safe fallback
        # values instead of crashing the entire application.

        return {
            "temp": 26.5,
            "humidity": 65.0,
            "weather_code": 0,
            "latitude": None,
            "longitude": None,
            "location_name": city_clean
        }


# ============================================================
# 2. RANDOM FOREST TRAINING DATA
# ============================================================

# Features:
# [Temperature, Humidity]

X_train = [
    [22.0, 50.0],
    [25.0, 55.0],
    [35.0, 40.0],
    [28.0, 65.0],
    [30.0, 70.0],
    [32.0, 60.0],
    [26.0, 82.0],
    [29.0, 86.0],
    [23.0, 92.0],
    [33.0, 95.0]
]


# 0 = Low Risk
# 1 = Medium Risk
# 2 = High Risk

y_train = [
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2,
    2
]


# ============================================================
# 3. CREATE AND TRAIN RANDOM FOREST
# ============================================================

ml_risk_model = RandomForestClassifier(
    n_estimators=20,
    random_state=42
)

ml_risk_model.fit(
    X_train,
    y_train
)


# ============================================================
# 4. RISK LABELS
# ============================================================

risk_labels = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}


# ============================================================
# 5. DISEASE-SPECIFIC ACTION ADVICE
# ============================================================

disease_advice = {

    "bacterial_spot": {
        "action":
            "Inspect leaves regularly and remove severely affected leaves.",
        "warning":
            "Avoid unnecessary leaf wetness and maintain good field hygiene."
    },

    "early_blight": {
        "action":
            "Remove severely affected leaves and monitor nearby plants.",
        "warning":
            "Avoid prolonged leaf moisture and improve air circulation."
    },

    "late_blight": {
        "action":
            "Closely inspect leaves and remove severely affected plant parts.",
        "warning":
            "High humidity and prolonged leaf wetness can increase disease spread."
    },

    "leaf_mold": {
        "action":
            "Improve ventilation and monitor the lower surfaces of leaves.",
        "warning":
            "Avoid excessive humidity and prolonged leaf wetness."
    },

    "septoria_leaf_spot": {
        "action":
            "Remove severely affected leaves and keep the field clean.",
        "warning":
            "Avoid overhead watering and prolonged leaf wetness."
    },

    "spider_mites": {
        "action":
            "Inspect leaves carefully, especially their lower surfaces.",
        "warning":
            "Monitor the crop regularly for increasing mite activity."
    },

    "target_spot": {
        "action":
            "Remove severely affected leaves and improve crop ventilation.",
        "warning":
            "Avoid prolonged leaf moisture."
    },

    "tomato_yellow_leaf_curl_virus": {
        "action":
            "Inspect plants for further spread and monitor insect activity.",
        "warning":
            "Remove severely affected plants according to local agricultural guidance."
    },

    "tomato_mosaic_virus": {
        "action":
            "Isolate suspected infected plants and maintain good field hygiene.",
        "warning":
            "Avoid spreading plant sap between healthy and infected plants."
    },

    "healthy": {
        "action":
            "Continue regular crop monitoring.",
        "warning":
            "Maintain normal field hygiene and monitor environmental conditions."
    }
}


# ============================================================
# 6. CROP RECOMMENDATIONS
# ============================================================

crop_database = {

    "cotton": {

        "High Risk":
            "CRITICAL ALERT: High humidity and warm temperature "
            "conditions may increase cotton pest and disease risk. "
            "Closely monitor the crop and take preventive action.",

        "Medium Risk":
            "WARNING: Moderate environmental risk detected. "
            "Monitor lower leaf surfaces and regularly inspect "
            "the crop for sucking pests.",

        "Low Risk":
            "EXCELLENT CONDITIONS: Environment is currently "
            "favorable for healthy cotton crop development. "
            "Continue normal field monitoring."
    },

    "tomato": {

        "High Risk":
            "CRITICAL ALERT: High humidity with cool or moderate "
            "temperature can increase tomato fungal disease risk. "
            "Closely monitor leaves and improve field ventilation.",

        "Medium Risk":
            "WARNING: Conditions may favor early blight or target "
            "spot. Monitor the crop regularly and avoid unnecessary "
            "leaf moisture.",

        "Low Risk":
            "HEALTHY ENVIRONMENT: Current climate conditions are "
            "relatively safe for tomato crops. Continue regular "
            "monitoring."
    }
}


# ============================================================
# 7. MAIN RISK PREDICTION FUNCTION
# ============================================================

def predict_risk(city: str, crop: str):

    # --------------------------------------------------------
    # GET REAL WEATHER
    # --------------------------------------------------------

    weather = get_weather(city)

    temperature = weather["temp"]
    humidity = weather["humidity"]

    # --------------------------------------------------------
    # RANDOM FOREST PREDICTION
    # --------------------------------------------------------

    predicted_code = ml_risk_model.predict(
        [[temperature, humidity]]
    )[0]

    final_risk = risk_labels[
        predicted_code
    ]

    # --------------------------------------------------------
    # RISK PROBABILITY
    # --------------------------------------------------------

    probabilities = ml_risk_model.predict_proba(
        [[temperature, humidity]]
    )[0]

    risk_score = float(
        probabilities[predicted_code]
    )

    # --------------------------------------------------------
    # CLEAN CROP NAME
    # --------------------------------------------------------

    crop_clean = crop.strip().lower()

    # --------------------------------------------------------
    # CROP-SPECIFIC RECOMMENDATION
    # --------------------------------------------------------

    if crop_clean in crop_database:

        recommendation = crop_database[
            crop_clean
        ][final_risk]

    else:

        if final_risk == "High Risk":

            recommendation = (
                f"ALERT for {crop_clean.capitalize()}: "
                f"High environmental risk detected. "
                f"Closely monitor the crop."
            )

        elif final_risk == "Medium Risk":

            recommendation = (
                f"WARNING for {crop_clean.capitalize()}: "
                f"Moderate environmental risk detected. "
                f"Regularly inspect the crop."
            )

        else:

            recommendation = (
                f"SAFE for {crop_clean.capitalize()}: "
                f"Current environmental conditions are "
                f"relatively favorable."
            )

    # --------------------------------------------------------
    # RETURN COMPLETE WEATHER + RISK DATA
    # --------------------------------------------------------

    return {

        "city":
            weather["location_name"],

        "crop":
            crop_clean.capitalize(),

        "temperature":
            temperature,

        "humidity":
            humidity,

        "weather_code":
            weather["weather_code"],

        "latitude":
            weather["latitude"],

        "longitude":
            weather["longitude"],

        "risk_level":
            final_risk,

        "risk_score":
            round(
                risk_score,
                4
            ),

        "recommendation":
            recommendation
    }