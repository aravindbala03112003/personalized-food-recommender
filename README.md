# Personalized Food Recommender (DINE MATE) 🍽️

An AI-powered, context-aware personalized food recommendation system built with **Streamlit**, **Machine Learning**, and **Nutritional Science**. 

DINE MATE provides holistic meal planning and recipe recommendations tailored to individual health conditions, allergies, dietary goals, climate conditions, and budget constraints.

---

## 🌟 Key Features

1. **AI Recommendation Engine**:
   - Hybrid filtering combining Content-Based Filtering (TF-IDF + Cosine Similarity) and Collaborative Filtering (SVD Matrix Factorization).
   - Popularity-based cold-start fallbacks and neural embedding support.
2. **Health & Medical Intelligence**:
   - Condition-aware dietary tailoring for Diabetes, Hypertension, Celiac Disease, IBS, Acid Reflux, Anemia, Arthritis, and High Cholesterol.
   - Strict allergen safety filtering (Peanuts, Tree nuts, Shellfish, Dairy, Gluten, Soy, Sesame, Sulfites).
3. **Climate & Context Optimization**:
   - Dynamic food recommendations tailored to Hot, Cold, Rainy, Humid, and Moderate climates.
4. **Budget & Meal Optimization**:
   - Per-meal and daily budget constraints with linear programming-style price/calorie/protein optimization.
5. **Goal Tracking & Analytics**:
   - Real-time macro-nutrient tracking (Protein, Carbs, Fat, Fiber) with BMI & caloric target progress.
6. **Authentication & User Management**:
   - Multi-role secure login and registration (User / Admin) powered by SQLite with SHA-256 hashing.
7. **Interactive Dashboards**:
   - Plotly-powered interactive visualizations for macronutrient distributions, review analytics, and model performance.

---

## 🚀 Quickstart (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/personalized-food-recommender.git
cd personalized-food-recommender
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application
```bash
python -m streamlit run food_recommender.py
```
Open your browser at `http://localhost:8501`.

---

## 🔐 Default Credentials (for testing)

- **Admin Account**: 
  - Username: `admin`
  - Password: `ADMIN2024`
- **Standard User**:
  - Username: `demo`
  - Password: `demo123`
- *New user registration is also available directly on the login screen.*

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:
```bash
pytest tests/
```

To run baseline model evaluation:
```bash
python eval_baseline_light.py
```

---

## ☁️ Zero-Cost Cloud Deployment (Streamlit Community Cloud)

1. Fork or push this repository to GitHub as `personalized-food-recommender`.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app** and configure:
   - **Repository**: `YOUR_USERNAME/personalized-food-recommender`
   - **Branch**: `main`
   - **Main file path**: `food_recommender.py`
4. Click **Deploy!**
