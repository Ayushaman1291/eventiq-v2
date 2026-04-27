"""
EventIQ v2 — Flask Backend
Model: Gradient Boosting Classifier
Endpoints: /health /metadata /predict /analyze /history /stats
"""
import os, sys, sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DB_PATH   = os.path.join(BASE_DIR, 'predictions.db')

sys.path.insert(0, BASE_DIR)
from utils.preprocess import preprocess_input, get_label_classes

app = Flask(__name__)
CORS(app)

# ── Load artifacts once at startup ────────────────────────────────────────────
print("Loading model artifacts...")
model    = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
meta     = joblib.load(os.path.join(MODEL_DIR, 'meta.pkl'))
print(f"  Model : {meta['model_name']}")
print(f"  Acc   : {meta['metrics']['accuracy']}")
print(f"  AUC   : {meta['metrics']['auc']}")

# ── SQLite init ───────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT,
            age              REAL,
            gender           TEXT,
            location         TEXT,
            event_type       TEXT,
            previous_events  REAL,
            income           REAL,
            event_rating     REAL,
            distance_km      REAL,
            organizer_score  REAL,
            social_buzz      REAL,
            registration_date TEXT,
            event_date       TEXT,
            prediction       INTEGER,
            probability      REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

REQUIRED = [
    'age', 'gender', 'location', 'event_type', 'previous_events',
    'income', 'event_rating', 'distance_km', 'organizer_score',
    'social_buzz', 'registration_date', 'event_date'
]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model':        meta['model_name'],
        'accuracy':     meta['metrics']['accuracy'],
        'dataset_size': meta['dataset_size']
    })


@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        'metrics':          meta['metrics'],
        'importances':      meta['importances'],
        'label_classes':    get_label_classes(encoders),
        'confusion_matrix': meta['confusion_matrix'],
        'class_report':     meta['class_report'],
        'model_name':       meta['model_name'],
        'n_features':       meta['n_features'],
        'dataset_size':     meta['dataset_size'],
    })


@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json(force=True)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        X    = preprocess_input(data, encoders, scaler, meta['feature_cols'])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO predictions
            (created_at, age, gender, location, event_type, previous_events,
             income, event_rating, distance_km, organizer_score, social_buzz,
             registration_date, event_date, prediction, probability)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            data['age'], data['gender'], data['location'], data['event_type'],
            data['previous_events'], data['income'], data['event_rating'],
            data['distance_km'], data['organizer_score'], data['social_buzz'],
            data['registration_date'], data['event_date'],
            pred, round(prob, 4)
        ))
        conn.commit()
        conn.close()

        return jsonify({
            'prediction':  pred,
            'probability': round(prob, 4),
            'label':       'Will Attend' if pred == 1 else 'Will Not Attend',
            'confidence':  f"{prob*100:.1f}%" if pred == 1
                           else f"{(1-prob)*100:.1f}%",
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(request.files['file'])
    except Exception as e:
        return jsonify({'error': f'Cannot read CSV: {e}'}), 400

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        return jsonify({'error': f'CSV missing columns: {missing_cols}'}), 400

    results = []
    for _, row in df.iterrows():
        try:
            X    = preprocess_input(row.to_dict(), encoders, scaler,
                                     meta['feature_cols'])
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            results.append({
                **{k: row.get(k) for k in REQUIRED},
                'prediction':  pred,
                'probability': round(prob, 4),
                'label':       'Will Attend' if pred == 1 else 'Will Not Attend'
            })
        except Exception:
            results.append({
                **row.to_dict(),
                'prediction':  None,
                'probability': None,
                'label':       'Error'
            })

    total     = len(results)
    attending = sum(1 for r in results if r['prediction'] == 1)
    return jsonify({
        'total':            total,
        'predicted_attend': attending,
        'predicted_skip':   total - attending,
        'attendance_rate':  round(attending / total * 100, 1) if total else 0,
        'rows':             results,
    })


@app.route('/history', methods=['GET'])
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT 100'
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/stats', methods=['GET'])
def stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total    = conn.execute(
        'SELECT COUNT(*) as n FROM predictions').fetchone()['n']
    attend   = conn.execute(
        'SELECT COUNT(*) as n FROM predictions WHERE prediction=1'
    ).fetchone()['n']
    by_type  = conn.execute("""
        SELECT event_type,
               COUNT(*) as total,
               SUM(prediction) as attending
        FROM predictions GROUP BY event_type
    """).fetchall()
    by_loc   = conn.execute("""
        SELECT location,
               COUNT(*) as total,
               SUM(prediction) as attending
        FROM predictions GROUP BY location
    """).fetchall()
    avg_prob = conn.execute(
        'SELECT AVG(probability) as ap FROM predictions'
    ).fetchone()['ap']
    conn.close()

    return jsonify({
        'total_predictions': total,
        'total_attending':   attend,
        'total_skipping':    total - attend,
        'attendance_rate':   round(attend / total * 100, 1) if total else 0,
        'avg_probability':   round(avg_prob or 0, 3),
        'by_event_type':     [dict(r) for r in by_type],
        'by_location':       [dict(r) for r in by_loc],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)