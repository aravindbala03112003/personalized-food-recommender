# User Guide - DINE MATE

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding Your Profile](#understanding-your-profile)
3. [Using the Web Interface](#using-the-web-interface)
4. [Interpreting Recommendations](#interpreting-recommendations)
5. [API Usage](#api-usage)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Getting Started

### What is DINE MATE?
The DINE MATE application is an AI-powered application that provides personalized food recommendations based on your:
- Health conditions (diabetes, hypertension, allergies)
- Climate and weather preferences
- Stress levels and mood
- Budget constraints
- Health goals (weight loss, muscle gain, general health)
- Activity level and lifestyle

### Quick Start (5 Minutes)
1. **Access the Application**
   - Web Interface: Open your browser to `http://localhost:8501`
   - API Access: Use `http://localhost:5000/api` endpoints

2. **Complete Your Profile**
   - Enter basic information (age, gender)
   - Select health conditions
   - Set your budget per meal
   - Choose your health goals

3. **Get Recommendations**
   - Click "GET PERSONALIZED RECOMMENDATIONS"
   - Explore different tabs (Health, Budget, Goals, Analysis)
   - Review food suggestions and explanations

## Understanding Your Profile

### Health Conditions
**Diabetes**: The system recommends low glycemic index foods and avoids high-sugar options.
- ✅ **Good choices**: Vegetables, lean proteins, complex carbs
- ❌ **Avoid**: Refined sugars, white bread, processed foods

**Hypertension**: Focus on low-sodium, potassium-rich foods.
- ✅ **Good choices**: Fresh fruits, vegetables, low-sodium options
- ❌ **Avoid**: Processed foods, high-sodium items

**Heart Disease**: Heart-healthy foods with omega-3 fatty acids.
- ✅ **Good choices**: Fish, nuts, olive oil, fiber-rich foods
- ❌ **Avoid**: Trans fats, excessive saturated fats

**Food Allergies**: The system automatically filters out allergens.
- **Nut Allergies**: All nut-containing foods are excluded
- **Dairy**: Lactose-free alternatives are prioritized
- **Gluten**: Gluten-free options are recommended

### Climate Preferences
**Hot Climate**: 
- Cooling foods (cucumber, watermelon, leafy greens)
- Hydrating options
- Light, easy-to-digest meals

**Cold Climate**:
- Warming foods (soups, stews, warm beverages)
- Higher calorie density
- Comfort foods that boost energy

### Stress Level (1-10 Scale)
**Low Stress (1-3)**: Regular balanced nutrition recommendations
**Moderate Stress (4-7)**: Include some comfort foods and stress-reducing nutrients
**High Stress (8-10)**: Focus on mood-boosting foods, magnesium-rich options

### Budget Constraints
- **Under $5/meal**: Basic, affordable staples with high nutritional value
- **$5-10/meal**: Balanced options with some variety
- **$10+/meal**: Premium ingredients and diverse choices

### Health Goals
**Weight Loss**:
- Low-calorie, high-satiety foods
- Portion control guidance
- Fiber-rich options

**Muscle Gain**:
- High-protein foods
- Post-workout nutrition
- Calorie-dense healthy options

**General Health**:
- Balanced macronutrients
- Variety of food groups
- Micronutrient optimization

## Using the Web Interface

### Main Dashboard
The interface consists of four main tabs:

#### 1. Health Recommendations Tab
- **Purpose**: Get recommendations based on your health conditions
- **Features**:
  - Top 5 health-optimized food suggestions
  - Health score for each recommendation
  - Reasoning for each suggestion
  - Nutritional information
- **How to Use**:
  1. Review the recommended foods
  2. Click on any food for detailed information
  3. Note the "Why recommended" explanations

#### 2. Budget Optimizer Tab
- **Purpose**: Find the best nutritional value within your budget
- **Features**:
  - Cost per serving information
  - Nutrition per dollar calculations
  - Daily meal planning estimates
  - Budget-friendly alternatives
- **How to Use**:
  1. Check the cost breakdown
  2. Compare nutrition per dollar values
  3. Plan your daily meals within budget
  4. Look for money-saving tips

#### 3. Goal Tracker Tab
- **Purpose**: Track progress toward your health goals
- **Features**:
  - Goal-specific food rankings
  - Progress predictions
  - Portion control guidance
  - Achievement tracking
- **How to Use**:
  1. Review goal alignment scores
  2. Follow portion size recommendations
  3. Track your compliance rate
  4. Adjust goals as needed

#### 4. Nutritional Analysis Tab
- **Purpose**: Understand the nutritional science behind recommendations
- **Features**:
  - Daily nutritional requirements
  - Macro and micronutrient breakdowns
  - Interactive charts and visualizations
  - Feature importance analysis
- **How to Use**:
  1. Review your daily nutrition targets
  2. Understand nutrient distributions
  3. Learn from feature importance charts
  4. Track nutritional balance

### Interactive Features

#### Adjusting Your Profile
- **Real-time Updates**: Changes to your profile immediately update recommendations
- **Sliders**: Use sliders for stress level, budget, and other numeric inputs
- **Dropdowns**: Select from predefined options for health conditions and goals
- **Multi-select**: Choose multiple allergies or dietary restrictions

#### Exploring Recommendations
- **Sorting**: Sort recommendations by health score, cost, or goal alignment
- **Filtering**: Filter by food category, preparation time, or dietary restrictions
- **Details**: Click on any food item for detailed nutritional information
- **Explanations**: Understand why each food was recommended

## Interpreting Recommendations

### Understanding Scores

#### Health Score (0.0 - 1.0)
- **0.8-1.0**: Excellent match for your health profile
- **0.6-0.8**: Good option with some benefits
- **0.4-0.6**: Moderate compatibility
- **Below 0.4**: May not be optimal for your health conditions

#### Goal Alignment (0.0 - 1.0)
- **0.9-1.0**: Perfect for your specific goal
- **0.7-0.9**: Very helpful for goal achievement
- **0.5-0.7**: Somewhat beneficial
- **Below 0.5**: Neutral or potentially counterproductive

#### Nutrition per Dollar
- **Higher values indicate better nutritional value for money**
- Compare foods within the same category
- Consider both protein and fiber content
- Factor in satiety and meal satisfaction

### Reading Explanations
Each recommendation includes reasoning such as:
- **"diabetes-friendly, low glycemic"**: Suitable for blood sugar management
- **"high protein value"**: Good for muscle building or weight management
- **"cooling effect for hot weather"**: Climate-appropriate choice
- **"fits your budget"**: Cost-effective option
- **"very filling"**: High satiety to prevent overeating

### Red Flags to Watch For
- ❌ Foods containing your allergens (shouldn't appear)
- ❌ Very expensive items exceeding your budget
- ❌ Foods contradicting your health conditions
- ❌ Extremely low scores across all categories

## API Usage

### Authentication
Currently no authentication required for local development.

### Base URL
```
http://localhost:5000/api
```

### Example API Calls

#### Get Health Recommendations
```python
import requests

response = requests.post('http://localhost:5000/api/recommendations/health', 
                        json={
                            'age': 35,
                            'has_diabetes': 1,
                            'climate_hot': 1,
                            'stress_level': 7,
                            'budget_per_meal': 8.0,
                            'n_recommendations': 10
                        })

recommendations = response.json()['recommendations']
for rec in recommendations:
    print(f"{rec['food_name']}: {rec['reasoning']}")
```

#### Get Budget Recommendations
```python
response = requests.post('http://localhost:5000/api/recommendations/budget', 
                        json={
                            'daily_budget': 15.0,
                            'target_calories': 2000,
                            'target_protein': 120,
                            'weight_loss_goal': True
                        })

budget_recs = response.json()['recommendations']
for rec in budget_recs:
    print(f"{rec['food_name']}: ${rec['cost_per_serving']:.2f}")
```

### Error Handling
The API returns appropriate HTTP status codes:
- **200**: Success
- **400**: Bad request (invalid input)
- **404**: Endpoint not found
- **500**: Internal server error

## Troubleshooting

### Common Issues

#### "No recommendations returned"
**Possible Causes**:
- Very restrictive profile (multiple allergies + low budget)
- System overloaded or model not trained
- Invalid input parameters

**Solutions**:
1. Relax some constraints (increase budget, reduce restrictions)
2. Check system status at `/api/health`
3. Restart the application
4. Verify input format matches API documentation

#### "Recommendations don't seem relevant"
**Possible Causes**:
- Profile information incomplete or incorrect
- Models need retraining with more data
- Edge case not handled properly

**Solutions**:
1. Double-check your profile information
2. Try different combinations of preferences
3. Report the issue with specific profile details
4. Use different recommendation tabs for comparison

#### "Application is slow or unresponsive"
**Possible Causes**:
- High system load
- Models training in background
- Insufficient system resources

**Solutions**:
1. Wait for model training to complete (initial startup takes 1-2 minutes)
2. Reduce number of recommendations requested
3. Check system resource usage (RAM, CPU)
4. Restart the application if necessary

#### "API returns 500 errors"
**Possible Causes**:
- Models not properly initialized
- Invalid data format
- System configuration issues

**Solutions**:
1. Check API status: `curl http://localhost:5000/api/health`
2. Verify models are trained: `curl http://localhost:5000/api/models/status`
3. Check application logs for error details
4. Restart the API service

### Getting Help
1. **Check Logs**: Review application logs for error messages
2. **System Status**: Use health check endpoints to verify system state
3. **Documentation**: Refer to API documentation for correct usage
4. **Community**: Report issues on the project repository

## FAQ

### General Questions

**Q: How accurate are the recommendations?**
A: The system achieves 85-96% accuracy across different models, but individual results may vary based on profile complexity and available data.

**Q: Can I use this for medical advice?**
A: No, this system provides general nutritional guidance only. Always consult healthcare professionals for medical advice.

**Q: How often should I update my profile?**
A: Update your profile when your health conditions, goals, or lifestyle change significantly.

### Technical Questions

**Q: What data does the system use?**
A: The system uses nutritional databases, user profile information, and synthetic training data. No personal data is stored permanently.

**Q: Can I integrate this with other applications?**
A: Yes, use the REST API endpoints for integration. Full API documentation is available.

**Q: Is my data secure?**
A: For local development, data stays on your machine. Production deployments should implement proper security measures.

### Usage Questions

**Q: Why do I get different recommendations each time?**
A: Some randomness is built into the models to provide variety. Core recommendations should remain consistent.

**Q: Can I save my favorite recommendations?**
A: The current version doesn't include favorites, but you can save the JSON responses from the API.

**Q: How do I report inaccurate recommendations?**
A: Document the specific profile settings and recommendations that seem incorrect, then report through the project repository.

## Tips for Best Results

1. **Be Honest**: Provide accurate profile information for best recommendations
2. **Be Specific**: Include all relevant health conditions and allergies
3. **Be Realistic**: Set achievable budget and goal expectations
4. **Be Patient**: Allow time for model training during first startup
5. **Be Flexible**: Try different settings to explore various options
6. **Be Consistent**: Update your profile regularly as circumstances change

## Advanced Features

### Batch Recommendations
For multiple users or meal planning, use the API to get recommendations for different profiles:
```python
profiles = [
    {'age': 25, 'goal': 'muscle_gain'},
    {'age': 45, 'has_diabetes': 1},
    {'age': 35, 'weight_loss_goal': 1}
]

for profile in profiles:
    recs = get_recommendations(profile)
    print(f"Recommendations for {profile}: {recs}")
```

### Custom Scoring
Combine scores from different models for personalized weighting:
```python
def custom_score(health_score, budget_score, goal_score):
    return (health_score * 0.5 + 
            budget_score * 0.3 + 
            goal_score * 0.2)
```

### Integration Examples
- **Meal Planning Apps**: Weekly meal generation based on user profiles
- **Grocery Apps**: Shopping list generation with budget optimization
- **Health Apps**: Nutrition tracking with personalized recommendations
- **Restaurant Apps**: Menu filtering based on dietary restrictions