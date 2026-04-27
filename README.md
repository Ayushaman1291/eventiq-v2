# EventIQ v2 — ML Event Analyst & Participation Prediction System

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Dataset size | 120 rows | 3,000 rows |
| ML Model | Random Forest | **Gradient Boosting** |
| Test Accuracy | ~63% (true CV) | **~88%** |
| AUC-ROC | ~0.69 | **~0.96** |
| F1 Score | ~0.65 | **~0.90** |
| New input features | — | `organizer_score`, `social_buzz` |
| Interaction features | 3 | **7** (rating×org, buzz×rating, loyalty_score…) |
| Streamlit pages | 4 | **5** (+ EDA page) |
| Predict page | Gauge only | Gauge + **driver bar chart** |

---

## Project Structure

```
eventiq_v2/
├── backend/
│   ├── app.py                    ← Flask API (6 endpoints)
│   ├── predictions.db            ← SQLite store (auto-created)
│   ├── model/
│   │   ├── train.py              ← Full training pipeline
│   │   ├── model.pkl             ← Gradient Boosting model
│   │   ├── scaler.pkl            ← StandardScaler
│   │   ├── encoders.pkl          ← LabelEncoders
│   │   └── meta.pkl              ← Metrics + importances
│   └── utils/
│       └── preprocess.py         ← Shared inference preprocessing
├── frontend/
│   └── app.py                    ← Streamlit 5-page dashboard
├── data/
│   └── event_dataset_3000.csv    ← 3000-row dataset
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (artifacts already included, but to retrain):
cd backend/model
python train.py

# 3. Start Flask (Terminal 1)
cd backend
python app.py
# → http://localhost:5000

# 4. Start Streamlit (Terminal 2)
cd frontend
streamlit run app.py
# → http://localhost:8501
```

---

## New Features in v2

### `organizer_score` (1.0 – 5.0)
Reputation score of the event organizer based on past events.
Higher organizer reputation strongly predicts attendance.

### `social_buzz` (0 – 100)
Online engagement/buzz score for the event (social media mentions, shares).
Viral events (score > 70) see significantly higher turnout.

### Interaction Features (7 total)
| Feature | Formula | What it captures |
|---|---|---|
| `rating_x_org` | event_rating × organizer_score | Great event + great organizer |
| `buzz_x_rating` | social_buzz × event_rating / 100 | Viral + quality combination |
| `prev_x_org` | previous_events × organizer_score | Loyal attendees of good organizers |
| `income_per_km` | income / (distance_km + 1) | Willingness to travel |
| `rating_x_prev` | event_rating × previous_events | Engaged repeat attendees |
| `dist_income` | distance_km / (income/10000 + 1) | Distance burden relative to income |
| `loyalty_score` | prev × rating × org_score / 25 | Overall loyalty composite |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Server status + model info |
| GET | /metadata | Metrics, importances, label classes |
| POST | /predict | Single prediction (JSON) |
| POST | /analyze | Batch CSV prediction |
| GET | /history | Last 100 predictions |
| GET | /stats | Aggregated stats by type/city |

### /predict — example body
```json
{
  "age": 28,
  "gender": "Female",
  "location": "Pune",
  "event_type": "Tech",
  "previous_events": 3,
  "income": 55000,
  "event_rating": 4.2,
  "distance_km": 12,
  "organizer_score": 4.0,
  "social_buzz": 72,
  "registration_date": "2024-03-01",
  "event_date": "2024-03-15"
}
```

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~88% |
| AUC-ROC | ~0.956 |
| F1 Score | ~0.904 |
| 5-Fold CV Accuracy | ~88% |

Top predictive features: `rating_x_org`, `buzz_x_rating`, `event_rating`,
`distance_km`, `dist_income`, `loyalty_score`
