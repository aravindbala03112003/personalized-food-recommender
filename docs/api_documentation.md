# 📂 API Documentation - DINE MATE

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently no authentication required for local development.

---

## Endpoints

### Health Check
**GET** `/health`

Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "message": "DINE MATE API is running",
  "models_loaded": true
}
```

---

### Health Recommendations
**POST** `/recommendations/health`

Get personalized health-aware food recommendations.

**Request Body:**
```json
{
  "age": 35,
  "has_diabetes": 1,
  "has_hypertension": 0,
  "climate_hot": 1,
  "stress_level": 7,
  "has_nut_allergy": 1,
  "budget_per_meal": 5.0,
  "weight_loss_goal": 1,
  "n_recommendations": 10
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "food_name": "Cucumber",
      "probability": 0.95,
      "calories_per_100g": 16,
      "glycemic_index": 10,
      "estimated_cost": 0.50,
      "diabetes_friendly": true,
      "good_for_hot_climate": true,
      "stress_relief": false,
      "reasoning": "diabetes-friendly, cooling effect for hot weather"
    }
  ],
  "user_profile": {...}
}
```

---

### Budget Recommendations
**POST** `/recommendations/budget`

Get budget-optimized meal recommendations.

**Request Body:**
```json
{
  "daily_budget": 10.0,
  "target_calories": 2000,
  "target_protein": 100,
  "weight_loss_goal": false,
  "n_recommendations": 10
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "food_name": "Lentil Soup",
      "cost_per_serving": 1.20,
      "nutrition_per_dollar": 8.5,
      "protein_per_dollar": 7.5,
      "calories_per_dollar": 116.7,
      "calories_per_serving": 140,
      "protein_per_serving": 9.0,
      "reasoning": "high protein, very affordable, diabetes-friendly"
    }
  ],
  "parameters": {...}
}
```

---

### Goal Recommendations
**POST** `/recommendations/goals`

Get recommendations aligned with health goals.

**Request Body:**
```json
{
  "age": 28,
  "gender": "female",
  "current_weight": 150,
  "weight_to_lose": 15,
  "weight_to_gain": 0,
  "activity_level": "moderate",
  "goal": "weight_loss",
  "weeks_on_plan": 4,
  "compliance_rate": 0.8,
  "n_recommendations": 10
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "food_name": "Cucumber Slices",
      "recommendation_probability": 0.92,
      "predicted_progress_impact": 8.5,
      "calories_per_portion": 8,
      "protein_per_portion": 0.3,
      "weight_loss_score": 9,
      "muscle_gain_score": 2,
      "portion_size": "unlimited",
      "goal_alignment": 0.90,
      "reasoning": "excellent for weight loss, very low calorie"
    }
  ],
  "portion_tips": [
    "Use smaller plates to make portions appear larger",
    "Fill half your plate with non-starchy vegetables"
  ],
  "user_profile": {...}
}
```

---

### Comprehensive Recommendations
**POST** `/recommendations/all`

Get recommendations from all models in a single request.

**Request Body:**
```json
{
  "age": 30,
  "has_diabetes": 1,
  "climate_hot": 1,
  "stress_level": 6,
  "has_nut_allergy": 0,
  "budget_per_meal": 8.0,
  "weight_loss_goal": 1,
  "gender": "female",
  "current_weight": 160,
  "weight_to_lose": 10,
  "activity_level": "moderate",
  "goal": "weight_loss",
  "weeks_on_plan": 2,
  "compliance_rate": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": {
    "health": [...],
    "budget": [...],
    "goals": [...],
    "tips": [...]
  }
}
```

---

### Model Status
**GET** `/models/status`

Check the status of all ML models.

**Response:**
```json
{
  "health_model": {
    "loaded": true,
    "trained": true
  },
  "budget_model": {
    "loaded": true,
    "trained": true
  },
  "goal_model": {
    "loaded": true,
    "trained": true
  },
  "hybrid_model": {
    "loaded": true,
    "trained": false
  }
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid input parameters"
}
```

### 404 Not Found
```json
{
  "error": "Endpoint not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Model not initialized"
}
```

---

## Usage Examples

### Python
```python
import requests

# Health recommendations
response = requests.post('http://localhost:5000/api/recommendations/health', 
                        json={
                            'age': 35,
                            'has_diabetes': 1,
                            'climate_hot': 1,
                            'stress_level': 7
                        })

recommendations = response.json()['recommendations']
for rec in recommendations:
    print(f"{rec['food_name']}: {rec['reasoning']}")
```

### JavaScript
```javascript
// Budget recommendations
const response = await fetch('http://localhost:5000/api/recommendations/budget', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    daily_budget: 10.0,
    target_calories: 2000,
    weight_loss_goal: true
  })
});

const data = await response.json();
console.log(data.recommendations);
```

### cURL
```bash
# Goal recommendations
curl -X POST http://localhost:5000/api/recommendations/goals \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "gender": "female",
    "goal": "weight_loss",
    "current_weight": 150,
    "weight_to_lose": 15
  }'
```

---

## Running the API

1. **Install dependencies:**
   ```bash
   pip install flask flask-cors
   ```

2. **Start the API server:**
   ```bash
   python flask_api.py
   ```

3. **API will be available at:**
   ```
   http://localhost:5000
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:5000/api/health
   ```

---

## Notes

- All ML models are initialized and trained on startup (takes 1-2 minutes)
- The API uses synthetic data for training if no real dataset is provided
- Models automatically handle missing parameters with sensible defaults
- All recommendations are personalized based on user profile
- Cost values are in USD ($)
- Calorie values are per serving/portion as specified
- Probability and alignment scores range from 0.0 to 1.0

---

## Development

To extend the API with new endpoints:

1. Add new route function in `flask_api.py`
2. Update this documentation
3. Add corresponding tests in `test_api.py`
4. Restart the API server

For production deployment, consider:
- Adding authentication/authorization
- Rate limiting
- Request validation
- Logging and monitoring
- Database integration for user preferences