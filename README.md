# Employee Attrition Risk Prediction System

Loging = admin pword= sPkuL5ezZfXEzhA

An AI-powered HR dashboard that predicts employee attrition risk using Random Forest and explains predictions using SHAP (SHapley Additive exPlanations).

## Tech Stack

- **Backend:** Python 3.12, Flask 3.1.3
- **ML/AI:** scikit-learn 1.8.0, SHAP 0.51.0, Pandas 3.0.1, NumPy 2.4.3
- **Database:** MySQL 8.0
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js 4.4.0

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/wickramasinghachenura-wq/FYPIIT.git
cd FYPIIT
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Import the database

Open MySQL Workbench, then:

1. **File > Open SQL Script** and select `db1.sql`
2. Click the lightning bolt to execute

This creates the `employee_risk_ai` database with all 7 tables and sample data (1,306 employees).

### 4. Create admin user

Replace `YOUR_PASSWORD` with your MySQL root password:

```bash
python -c "
import mysql.connector, bcrypt
conn = mysql.connector.connect(host='localhost', user='root', password='YOUR_PASSWORD', database='employee_risk_ai')
cur = conn.cursor()
cur.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', ('admin', bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode()))
conn.commit()
conn.close()
print('Admin user created')
"
```

### 5. Configure database connection

Open `app.py` and update the MySQL password in the database configuration section to match your local setup.

### 6. Run the application

```bash
python app.py
```

### 7. Open in browser

Navigate to `http://localhost:5000`

Login with:
- **Username:** admin
- **Password:** admin123

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Flask backend with 8 REST API endpoints |
| `index.html` | Main dashboard (single-page application) |
| `login.html` | Login page |
| `rf_model.pkl` | Trained Random Forest model (200 trees) |
| `model_metrics.json` | Model performance metrics |
| `db1.sql` | MySQL database export (schema + data) |
| `fypbetaproject (1).ipynb` | Jupyter notebook for model training |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | IBM HR Employee Attrition dataset |
| `requirements.txt` | Python dependencies |

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80.2% |
| Precision | 73.7% |
| Recall | 53.2% |
| F1-Score | 61.8% |
| AUC-ROC | 90.4% |

## Features

- Risk prediction for individual employees (High/Low) with confidence scores
- SHAP-based explainability with colour-coded factor chips
- Plain-language interpretations and HR action recommendations
- Department risk breakdown chart
- Temporal trend charts (stress over time, hours worked)
- Search, filter, sort, and CSV export
- Glassmorphism dark UI design
