from sklearn.ensemble import RandomForestClassifier
import urllib.parse
import urllib.request
import json


# ============================================================
# WEATHER CODE -> HUMAN READABLE CONDITION
# ============================================================

def weather_code_to_condition(code):
    """
    Convert Open-Meteo WMO weather code into
    a simple human-readable condition.
    """

    weather_conditions = {

        0: "clear sky",

        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",

        45: "fog",
        48: "fog",

        51: "light drizzle",
        53: "moderate drizzle",
        55: "heavy drizzle",

        56: "freezing drizzle",
        57: "heavy freezing drizzle",

        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",

        66: "freezing rain",
        67: "heavy freezing rain",

        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",

        77: "snow grains",

        80: "light rain showers",
        81: "moderate rain showers",
        82: "heavy rain showers",

        85: "light snow showers",
        86: "heavy snow showers",

        95: "thunderstorm",

        96: "thunderstorm with hail",
        99: "heavy thunderstorm with hail"
    }

    return weather_conditions.get(
        code,
        "unknown"
    )


# ============================================================
# 1. REAL WEATHER API
# ============================================================

def get_weather(city):
    """
    Get REAL current weather + 7-day forecast
    using Open-Meteo.

    Returns:

        Current:
        - temperature
        - humidity
        - weather_code
        - condition
        - wind_kmh

        Location:
        - latitude
        - longitude
        - location_name

        Forecast:
        - date
        - high
        - low
        - rainfall_mm
        - weather_code
        - condition
    """

    city_clean = city.strip()

    if not city_clean:
        raise ValueError(
            "City name is required."
        )

    try:

        # ========================================================
        # STEP 1: CITY -> LATITUDE/LONGITUDE
        # ========================================================

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

        locations = geocode_data.get(
            "results",
            []
        )

        if not locations:

            raise ValueError(
                f"Could not find weather location "
                f"for '{city_clean}'."
            )

        location = locations[0]

        latitude = location["latitude"]

        longitude = location["longitude"]

        location_name = location.get(
            "name",
            city_clean
        )

        # ========================================================
        # STEP 2: GET CURRENT + 7-DAY WEATHER
        # ========================================================

        weather_params = urllib.parse.urlencode({

            "latitude": latitude,

            "longitude": longitude,

            # ----------------------------------------------------
            # CURRENT WEATHER
            # ----------------------------------------------------

            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m"
            ),

            # ----------------------------------------------------
            # DAILY FORECAST
            # ----------------------------------------------------

            "daily": (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_sum"
            ),

            "forecast_days": 7,

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

        # ========================================================
        # STEP 3: CURRENT WEATHER
        # ========================================================

        current = weather_data.get(
            "current",
            {}
        )

        temperature = float(
            current.get(
                "temperature_2m",
                0
            )
        )

        humidity = float(
            current.get(
                "relative_humidity_2m",
                0
            )
        )

        weather_code = int(
            current.get(
                "weather_code",
                0
            )
        )

        wind_kmh = float(
            current.get(
                "wind_speed_10m",
                0
            )
        )

        condition = weather_code_to_condition(
            weather_code
        )

        # ========================================================
        # STEP 4: 7-DAY FORECAST
        # ========================================================

        daily = weather_data.get(
            "daily",
            {}
        )

        dates = daily.get(
            "time",
            []
        )

        max_temperatures = daily.get(
            "temperature_2m_max",
            []
        )

        min_temperatures = daily.get(
            "temperature_2m_min",
            []
        )

        daily_weather_codes = daily.get(
            "weather_code",
            []
        )

        rainfall = daily.get(
            "precipitation_sum",
            []
        )

        forecast = []

        for i in range(len(dates)):

            day_code = int(
                daily_weather_codes[i]
            )

            forecast.append({

                "date": dates[i],

                "high": float(
                    max_temperatures[i]
                ),

                "low": float(
                    min_temperatures[i]
                ),

                "rainfall_mm": float(
                    rainfall[i]
                ),

                "weather_code": day_code,

                "condition":
                    weather_code_to_condition(
                        day_code
                    )
            })

        # ========================================================
        # STEP 5: RETURN WEATHER DATA
        # ========================================================

        return {

            # Current weather

            "temp":
                temperature,

            "humidity":
                humidity,

            "weather_code":
                weather_code,

            "condition":
                condition,

            "wind_kmh":
                wind_kmh,

            # Location

            "latitude":
                latitude,

            "longitude":
                longitude,

            "location_name":
                location_name,

            # Forecast

            "forecast":
                forecast
        }

    except Exception as e:

        print(
            f"Weather API error for "
            f"{city_clean}: {e}"
        )

        # ========================================================
        # FALLBACK
        # ========================================================

        return {

            "temp":
                26.5,

            "humidity":
                65.0,

            "weather_code":
                0,

            "condition":
                "clear sky",

            "wind_kmh":
                0.0,

            "latitude":
                None,

            "longitude":
                None,

            "location_name":
                city_clean,

            "forecast":
                []
        }


# ============================================================
# 2. RANDOM FOREST TRAINING DATA
# ============================================================

# Features:
#
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
            "Inspect leaves regularly and "
            "remove severely affected leaves.",

        "warning":
            "Avoid unnecessary leaf wetness "
            "and maintain good field hygiene."
    },

    "early_blight": {

        "action":
            "Remove severely affected leaves "
            "and monitor nearby plants.",

        "warning":
            "Avoid prolonged leaf moisture "
            "and improve air circulation."
    },

    "late_blight": {

        "action":
            "Closely inspect leaves and remove "
            "severely affected plant parts.",

        "warning":
            "High humidity and prolonged leaf "
            "wetness can increase disease spread."
    },

    "leaf_mold": {

        "action":
            "Improve ventilation and monitor "
            "the lower surfaces of leaves.",

        "warning":
            "Avoid excessive humidity and "
            "prolonged leaf wetness."
    },

    "septoria_leaf_spot": {

        "action":
            "Remove severely affected leaves "
            "and keep the field clean.",

        "warning":
            "Avoid overhead watering and "
            "prolonged leaf wetness."
    },

    "spider_mites": {

        "action":
            "Inspect leaves carefully, "
            "especially their lower surfaces.",

        "warning":
            "Monitor the crop regularly "
            "for increasing mite activity."
    },

    "target_spot": {

        "action":
            "Remove severely affected leaves "
            "and improve crop ventilation.",

        "warning":
            "Avoid prolonged leaf moisture."
    },

    "tomato_yellow_leaf_curl_virus": {

        "action":
            "Inspect plants for further spread "
            "and monitor insect activity.",

        "warning":
            "Remove severely affected plants "
            "according to local agricultural guidance."
    },

    "tomato_mosaic_virus": {

        "action":
            "Isolate suspected infected plants "
            "and maintain good field hygiene.",

        "warning":
            "Avoid spreading plant sap between "
            "healthy and infected plants."
    },

    "healthy": {

        "action":
            "Continue regular crop monitoring.",

        "warning":
            "Maintain normal field hygiene and "
            "monitor environmental conditions."
    }
}


# ============================================================
# 6. CROP RECOMMENDATIONS
# ============================================================

crop_database = {

    "cotton": {

        "High Risk":

            "CRITICAL ALERT: High humidity and "
            "warm temperature conditions may "
            "increase cotton pest and disease risk. "
            "Closely monitor the crop and take "
            "preventive action.",

        "Medium Risk":

            "WARNING: Moderate environmental risk "
            "detected. Monitor lower leaf surfaces "
            "and regularly inspect the crop for "
            "sucking pests.",

        "Low Risk":

            "EXCELLENT CONDITIONS: Environment is "
            "currently favorable for healthy cotton "
            "crop development. Continue normal "
            "field monitoring."
    },

    "tomato": {

        "High Risk":

            "CRITICAL ALERT: High humidity with "
            "cool or moderate temperature can "
            "increase tomato fungal disease risk. "
            "Closely monitor leaves and improve "
            "field ventilation.",

        "Medium Risk":

            "WARNING: Conditions may favor early "
            "blight or target spot. Monitor the "
            "crop regularly and avoid unnecessary "
            "leaf moisture.",

        "Low Risk":

            "HEALTHY ENVIRONMENT: Current climate "
            "conditions are relatively safe for "
            "tomato crops. Continue regular "
            "monitoring."
    }
}


# ============================================================
# 7. MAIN RISK PREDICTION FUNCTION
# ============================================================

def predict_risk(city: str, crop: str):

    # ========================================================
    # GET REAL WEATHER
    # ========================================================

    weather = get_weather(city)

    temperature = weather["temp"]

    humidity = weather["humidity"]

    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    predicted_code = ml_risk_model.predict(

        [[
            temperature,
            humidity
        ]]

    )[0]

    final_risk = risk_labels[
        predicted_code
    ]

    # ========================================================
    # RISK PROBABILITY
    # ========================================================

    probabilities = ml_risk_model.predict_proba(

        [[
            temperature,
            humidity
        ]]

    )[0]

    risk_score = float(

        probabilities[
            predicted_code
        ]

    )

    # ========================================================
    # CLEAN CROP NAME
    # ========================================================

    crop_clean = crop.strip().lower()

    # ========================================================
    # CROP-SPECIFIC RECOMMENDATION
    # ========================================================

    if crop_clean in crop_database:

        recommendation = crop_database[
            crop_clean
        ][
            final_risk
        ]

    else:

        if final_risk == "High Risk":

            recommendation = (

                f"ALERT for "
                f"{crop_clean.capitalize()}: "

                f"High environmental risk detected. "

                f"Closely monitor the crop."

            )

        elif final_risk == "Medium Risk":

            recommendation = (

                f"WARNING for "
                f"{crop_clean.capitalize()}: "

                f"Moderate environmental risk detected. "

                f"Regularly inspect the crop."

            )

        else:

            recommendation = (

                f"SAFE for "
                f"{crop_clean.capitalize()}: "

                f"Current environmental conditions "
                f"are relatively favorable."

            )

    # ========================================================
    # RETURN COMPLETE WEATHER + RISK DATA
    # ========================================================

    return {

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        "city":
            weather["location_name"],

        "latitude":
            weather["latitude"],

        "longitude":
            weather["longitude"],

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        "crop":
            crop_clean.capitalize(),

        # ----------------------------------------------------
        # CURRENT WEATHER
        # ----------------------------------------------------

        "temperature":
            temperature,

        "humidity":
            humidity,

        "wind_kmh":
            weather["wind_kmh"],

        "weather_code":
            weather["weather_code"],

        "condition":
            weather["condition"],

        # ----------------------------------------------------
        # 7-DAY FORECAST
        # ----------------------------------------------------

        "forecast":
            weather["forecast"],

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        "risk_level":
            final_risk,

        "risk_score":
            round(
                risk_score,
                4
            ),

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        "recommendation":
            recommendation
    }