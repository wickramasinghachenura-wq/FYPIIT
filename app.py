from flask import Flask, jsonify, request, session, redirect, send_from_directory
from flask_cors import CORS
from mysql.connector import pooling, Error as MySQLError
from dotenv import load_dotenv
import pandas as pd
import joblib
import json
import shap          # for explaining model predictions
import bcrypt        # for hashing passwords
import os

load_dotenv()  # loads db credentials etc from .env file

# ----------------- APP SETUP -----------------
app = Flask(__name__)
CORS(app)  # needed so frontend can call the API
app.secret_key = os.environ.get("SECRET_KEY")

# cookie security settings
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

# load the trained random forest model
print("Loading Model...")
model = joblib.load("rf_model.pkl")
FEATURE_COLUMNS = list(model.feature_names_in_)  # need same feature order as training

# threshold is 0.35 instead of the usual 0.5 so we catch more at-risk employees
_DEFAULT_THRESHOLD = 0.35
try:
    with open("model_metrics.json") as _mf:
        _saved = json.load(_mf)
    RISK_THRESHOLD = float(_saved.get("threshold", _DEFAULT_THRESHOLD))
except Exception:
    RISK_THRESHOLD = _DEFAULT_THRESHOLD
print(f"Risk threshold: {RISK_THRESHOLD}")

# shap explainer - this is what tells us WHY someone is high risk
print("Initializing SHAP Explainer...")
explainer = shap.TreeExplainer(model)
print("SHAP Initialized.")

# ----------------- CONFIGURATION -----------------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME", "employee_risk_ai")
}

# ----------------- CONNECTION POOLING -----------------
# connection pool so we dont keep opening and closing db connections
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="employee_risk_pool",
        pool_size=5,           # max 5 connections at once
        pool_reset_session=True,
        **DB_CONFIG
    )
    print("Database connection pool created successfully.")
except MySQLError as e:
    print(f"Error creating connection pool: {e}")
    connection_pool = None


def get_connection():
    """Get a connection from the pool with error handling."""
    if connection_pool is None:
        raise MySQLError("Connection pool not initialized")
    return connection_pool.get_connection()


# ----------------- FEATURE BUILDER -----------------
def build_features(employee_id=None):
    """
    Build features from database matching the exact query used during model training.
    Uses employment_records for job_satisfaction, work_life_balance, performance_score
    — same as the notebook training query.
    """
    conn = None
    try:
        conn = get_connection()

        where_clause = ""
        params = []
        if employee_id is not None:
            where_clause = "WHERE e.employee_id = %s"
            params = [employee_id]

        # ✅ FIX 1: Use the same query as the notebook — pulling from employment_records
        # NOT from survey_responses via the employee_features view
        query = f"""
            SELECT
                e.employee_id,
                e.age,
                e.department,
                AVG(er.performance_score)  AS performance_score,
                AVG(er.job_satisfaction)   AS job_satisfaction,
                AVG(er.work_life_balance)  AS work_life_balance,
                AVG(a.hours_worked)        AS avg_hours_worked,
                AVG(s.stress_level)        AS avg_stress_level
            FROM employees e
            LEFT JOIN employment_records er ON e.employee_id = er.employee_id
            LEFT JOIN attendance_logs a     ON e.employee_id = a.employee_id
            LEFT JOIN survey_responses s    ON e.employee_id = s.employee_id
            {where_clause}
            GROUP BY e.employee_id, e.age, e.department
        """

        df = pd.read_sql(query, conn, params=params if params else None)

    except MySQLError as e:
        print(f"Database error: {e}")
        raise
    finally:
        if conn is not None and conn.is_connected():
            conn.close()

    # Use sensible defaults for missing data instead of 0
    # (0 would make missing employees appear as lowest-risk, which is misleading)
    defaults = {
        "avg_stress_level":  3.0,   # mid-scale (1-5)
        "avg_hours_worked":  8.5,   # typical working day
        "job_satisfaction":  3.0,   # mid-scale (1-5)
        "work_life_balance": 3.0,   # mid-scale (1-5)
        "performance_score": 3.0,   # mid-scale (1-5)
        "age":               35.0,  # approximate median age
    }
    return df.fillna(defaults).fillna(0)


# ----------------- DETAILED RISK REPORT GENERATOR -----------------
def generate_risk_report(employee_id, risk_label, top_features, original_row):
    """
    Takes the top SHAP features and turns them into a readable report
    with risk factors and recommended actions for HR.
    """
    risk = "High" if risk_label == 1 else "Low"

    negative_factors = []   # things making the risk worse
    positive_factors = []   # things keeping risk down
    actions = []            # what HR should do about it

    # check each top feature against thresholds to categorize them
    for _, row in top_features.iterrows():
        feature = row["feature"]
        value = original_row.get(feature, 0)

        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if feature == "age":
            if value <= 25:
                negative_factors.append(f"employee is young ({int(value)} yrs) — younger employees tend to have higher turnover")
                actions.append("discuss long-term career growth and development opportunities")
            elif value >= 50:
                negative_factors.append(f"employee is senior ({int(value)} yrs) — may be considering early retirement")
                actions.append("explore retention incentives and role flexibility")
            else:
                positive_factors.append(f"employee age ({int(value)} yrs) is in a stable range")

        elif feature == "avg_hours_worked":
            if value >= 9.5:
                negative_factors.append(f"attendance shows excessive overtime ({value:.1f} hrs/day)")
                actions.append("reduce overtime and rebalance workload")
            elif value < 7:
                negative_factors.append(f"attendance shows unusually low hours ({value:.1f} hrs/day)")
                actions.append("investigate potential disengagement or underutilization")
            else:
                positive_factors.append(f"attendance shows healthy working hours ({value:.1f} hrs/day)")

        elif feature == "avg_stress_level":
            if value >= 4:
                negative_factors.append("stress score is critical")
                actions.append("initiate immediate stress management counseling")
            elif value >= 3:
                negative_factors.append("stress score is elevated")
                actions.append("schedule a wellbeing check-in")
            else:
                positive_factors.append("stress score is healthy")

        elif feature == "work_life_balance":
            if value <= 2:
                negative_factors.append("work-life balance score is poor")
                actions.append("offer flexible working arrangements")
            elif value == 3:
                negative_factors.append("work-life balance score is moderate")
            else:
                positive_factors.append("work-life balance score is strong")

        elif feature == "job_satisfaction":
            if value <= 2:
                negative_factors.append("job satisfaction score is low")
                actions.append("conduct a role alignment and career progression review")
            elif value >= 4:
                positive_factors.append("job satisfaction score is high")

        elif feature == "performance_score":
            if value <= 2:
                negative_factors.append("recorded performance score is low")
                actions.append("provide targeted performance coaching")
            elif value >= 4:
                positive_factors.append("recorded performance score is strong")

        else:
            shap_value = row.get("shap_value", 0)
            if shap_value > 0:
                negative_factors.append(f"{feature.replace('_', ' ')} is a risk factor")
            elif shap_value < 0:
                positive_factors.append(f"{feature.replace('_', ' ')} is a protective factor")

    if negative_factors:
        interpretation = "Key risk factors include: " + "; ".join(negative_factors) + "."
        if positive_factors:
            interpretation += " However, " + "; ".join(positive_factors) + "."
    elif positive_factors:
        interpretation = "The employee shows strong indicators: " + "; ".join(positive_factors) + "."
    else:
        interpretation = "The employee shows a mix of average indicators with no extreme outliers."

    actions = list(dict.fromkeys(actions))  # remove duplicates but keep order

    if not actions:
        actions = ["Review specific department benchmarks"] if risk == "High" else \
                  ["Continue routine monitoring and recognise strong performance"]

    return {
        "risk_level": risk,
        "interpretation": interpretation,
        "actions": actions,
        "negative_factors": negative_factors,
        "positive_factors": positive_factors
    }


# ----------------- AUTH HELPER -----------------
def login_required(f):
    """checks if user is logged in before letting them access the page."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            # For API calls return 401, for page requests redirect
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorised. Please log in."}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


# ----------------- AUTH ROUTES -----------------
@app.route("/login")
def login_page():
    """Serve the login page."""
    if session.get("user_id"):
        return redirect("/")
    return send_from_directory(".", "login.html")


@app.route("/")
@login_required
def index():
    """Serve the main dashboard (protected)."""
    return send_from_directory(".", "index.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    """Handle login — verify credentials and create session."""
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_id, username, password_hash FROM users WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()

        # check hashed password
        if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            return jsonify({"success": True, "username": user["username"]})
        else:
            return jsonify({"error": "Invalid username or password."}), 401

    except MySQLError as e:
        print(f"DB error in login: {e}")
        return jsonify({"error": "Database error occurred."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Clear session and log out."""
    session.clear()
    return jsonify({"success": True})


# ----------------- RISK API -----------------
@app.route("/api/risk/<int:employee_id>")
@login_required
def get_risk(employee_id):
    """
    Main prediction route - gets employee data, runs it through the model,
    explains the prediction with SHAP, and saves everything to the db.
    """

    if employee_id <= 0:
        return jsonify({"error": "Invalid employee ID"}), 400

    try:
        # 1. Build Data (matching training query exactly)
        raw_df = build_features(employee_id=employee_id)

        if raw_df.empty:
            return jsonify({"error": "Employee not found"}), 404

        # 2. one-hot encode department and make sure columns match what model expects
        df = pd.get_dummies(raw_df.drop(columns=["employee_id"]))
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # 3. get the probability of leaving and check against our threshold
        raw_prob = float(model.predict_proba(df)[0][1])
        risk = int(raw_prob >= RISK_THRESHOLD)
        prob = raw_prob

        # 4. SHAP - figure out which features pushed the risk up or down
        shap_values = explainer.shap_values(df.astype(float))

        # shap can return different formats depending on sklearn version, handle all of them
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # features we can actually explain in the report
        reportable_features = [
            "age",
            "avg_hours_worked",
            "avg_stress_level",
            "work_life_balance",
            "job_satisfaction",
            "performance_score",
        ]

        drivers_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "shap_value": sv
        })

        # for high risk: show features pushing risk UP (positive shap)
        # for low risk: show features keeping risk DOWN (negative shap)
        # then fill remaining slots with the other direction
        filtered = drivers_df[drivers_df["feature"].isin(reportable_features)]
        if risk == 1:
            primary = filtered[filtered["shap_value"] > 0].sort_values("shap_value", ascending=False)
            secondary = filtered[filtered["shap_value"] <= 0].reindex(
                filtered[filtered["shap_value"] <= 0]["shap_value"].abs().sort_values(ascending=False).index
            )
        else:
            primary = filtered[filtered["shap_value"] < 0].sort_values("shap_value")
            secondary = filtered[filtered["shap_value"] >= 0].sort_values("shap_value", ascending=False)
        top_drivers = pd.concat([primary, secondary]).head(4)

        # 5. Generate Report
        combined_row = raw_df.iloc[0].to_dict()
        report = generate_risk_report(employee_id, risk, top_drivers, combined_row)

        # 6. save prediction + shap values to db so we have a record of it
        persist_conn = None
        try:
            persist_conn = get_connection()
            cursor = persist_conn.cursor()

            cursor.execute(
                """INSERT INTO risk_predictions
                   (employee_id, risk_level, confidence, prediction_date)
                   VALUES (%s, %s, %s, NOW())""",
                (employee_id, report["risk_level"], round(prob * 100, 2))
            )
            prediction_id = cursor.lastrowid

            shap_rows = [
                (prediction_id, row["feature"], float(row["shap_value"]))
                for _, row in drivers_df.iterrows()
            ]
            cursor.executemany(
                """INSERT INTO explainability_results
                   (prediction_id, feature_name, shap_value)
                   VALUES (%s, %s, %s)""",
                shap_rows
            )

            persist_conn.commit()
        except Exception as e:
            print(f"Warning: could not persist prediction: {e}")
            if persist_conn:
                persist_conn.rollback()
        finally:
            if persist_conn and persist_conn.is_connected():
                persist_conn.close()

        return jsonify({
            "employee_id": employee_id,
            "risk_label": report["risk_level"],
            "confidence": round(prob * 100, 2),
            "interpretation": report["interpretation"],
            "actions": report["actions"],
            "negative_factors": report["negative_factors"],
            "positive_factors": report["positive_factors"]
        })

    except MySQLError as e:
        print(f"Database error in get_risk: {e}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        print(f"Unexpected error in get_risk: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# ----------------- TEMPORAL DATA API -----------------
@app.route("/api/temporal/<int:employee_id>")
@login_required
def get_temporal_data(employee_id):
    """gets attendance and survey history for the trend charts."""

    if employee_id <= 0:
        return jsonify({"error": "Invalid employee ID"}), 400

    conn = None
    try:
        conn = get_connection()

        attendance = pd.read_sql(
            """SELECT log_date, hours_worked
               FROM attendance_logs
               WHERE employee_id = %s
               ORDER BY log_date ASC""",
            conn,
            params=[employee_id]
        )

        survey = pd.read_sql(
            """SELECT survey_date, stress_level
               FROM survey_responses
               WHERE employee_id = %s
               ORDER BY survey_date ASC""",
            conn,
            params=[employee_id]
        )

        attendance["log_date"] = attendance["log_date"].astype(str)
        survey["survey_date"] = survey["survey_date"].astype(str)

        return jsonify({
            "attendance": attendance.to_dict(orient="records"),
            "survey": survey.to_dict(orient="records")
        })

    except MySQLError as e:
        print(f"Database error in get_temporal_data: {e}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        print(f"Unexpected error in get_temporal_data: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500
    finally:
        if conn is not None and conn.is_connected():
            conn.close()


# ----------------- CURRENT USER API -----------------
@app.route("/api/me")
@login_required
def get_me():
    """Return the logged-in user's username."""
    return jsonify({"username": session.get("username", "User")})


# ----------------- ALL EMPLOYEES API -----------------
@app.route("/api/employees")
@login_required
def get_all_employees():
    """runs prediction on all employees at once for the main table."""
    conn = None
    try:
        raw_df = build_features()
        if raw_df.empty:
            return jsonify([])

        conn = get_connection()
        emp_info = pd.read_sql(
            "SELECT employee_id, job_role FROM employees", conn
        )

        df = pd.get_dummies(raw_df.drop(columns=["employee_id"]))
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        probs = model.predict_proba(df)[:, 1].tolist()  # probability of leaving for each
        risks = [int(p >= RISK_THRESHOLD) for p in probs]

        emp_map = emp_info.set_index("employee_id")["job_role"].to_dict()

        results = []
        for idx, (_, row) in enumerate(raw_df.iterrows()):
            risk = risks[idx]
            raw_prob = probs[idx]
            confidence = raw_prob  # always show probability of leaving
            results.append({
                "employee_id": int(row["employee_id"]),
                "department": row["department"],
                "job_title": emp_map.get(int(row["employee_id"]), "N/A"),  # key stays job_title for frontend
                "age": int(row["age"]),
                "risk_label": "High" if risk == 1 else "Low",
                "confidence": round(confidence * 100, 2),
            })

        # High-risk employees first, then sorted by ID
        results.sort(key=lambda x: (0 if x["risk_label"] == "High" else 1, x["employee_id"]))
        return jsonify(results)

    except MySQLError as e:
        print(f"Database error in get_all_employees: {e}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        print(f"Unexpected error in get_all_employees: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500
    finally:
        if conn is not None and conn.is_connected():
            conn.close()


# ----------------- HEALTH CHECK API -----------------
@app.route("/api/health")
def health_check():
    """Health check endpoint for monitoring."""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "database_pool": connection_pool is not None
    }

    try:
        conn = get_connection()
        conn.close()
        status["database_connected"] = True
    except Exception:
        status["database_connected"] = False
        status["status"] = "degraded"

    return jsonify(status)


# ----------------- MODEL METRICS API -----------------
@app.route("/api/metrics")
@login_required
def get_metrics():
    """returns model metrics from the json file, not hardcoded."""
    try:
        with open("model_metrics.json") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "model_metrics.json not found — run retrain_ibm.py first"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------- ERROR HANDLERS -----------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ----------------- START SERVER -----------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
