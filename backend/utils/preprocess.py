import pandas as pd
import numpy as np


def preprocess_input(raw: dict, encoders: dict, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Convert a raw prediction request into a scaled feature DataFrame.

    New fields vs v1:
        organizer_score  (float 1.0–5.0)
        social_buzz      (int   0–100)
    """
    reg_date   = pd.to_datetime(raw['registration_date'])
    event_date = pd.to_datetime(raw['event_date'])

    days_to_event = (event_date - reg_date).days
    event_month   = event_date.month
    reg_month     = reg_date.month
    event_dow     = event_date.dayofweek
    is_weekend    = int(event_dow >= 5)
    event_quarter = (event_date.month - 1) // 3 + 1

    age    = float(raw['age'])
    inc    = float(raw['income'])
    rat    = float(raw['event_rating'])
    dist   = float(raw['distance_km'])
    prev   = float(raw['previous_events'])
    org    = float(raw['organizer_score'])
    buzz   = float(raw['social_buzz'])

    row = {
        'age':              age,
        'gender':           raw['gender'],
        'location':         raw['location'],
        'event_type':       raw['event_type'],
        'previous_events':  prev,
        'income':           inc,
        'event_rating':     rat,
        'distance_km':      dist,
        'organizer_score':  org,
        'social_buzz':      buzz,
        'days_to_event':    float(days_to_event),
        'event_month':      float(event_month),
        'reg_month':        float(reg_month),
        'event_dow':        float(event_dow),
        'is_weekend':       float(is_weekend),
        'event_quarter':    float(event_quarter),
        # Interaction features
        'rating_x_org':     rat * org,
        'buzz_x_rating':    buzz * rat / 100,
        'prev_x_org':       prev * org,
        'income_per_km':    inc / (dist + 1),
        'rating_x_prev':    rat * prev,
        'dist_income':      dist / (inc / 10000 + 1),
        'loyalty_score':    prev * rat * org / 25,
    }

    # Encode categoricals
    for col in ['gender', 'location', 'event_type']:
        le  = encoders[col]
        val = row[col]
        row[col] = float(
            le.transform([val])[0] if val in le.classes_
            else le.transform([le.classes_[0]])[0]
        )

    df = pd.DataFrame([row])[feature_cols]
    return pd.DataFrame(scaler.transform(df), columns=feature_cols)


def get_label_classes(encoders: dict) -> dict:
    return {col: list(le.classes_) for col, le in encoders.items()}
