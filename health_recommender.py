"""
Health-Aware Food Recommender Module for DINE MATE
Provides health-condition based recommendations using Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class HealthAwareFoodRecommender:
    """Recommends foods based on user health conditions, allergies, and climate."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_trained = False
        self.general_health_model = None
        self.food_db = self.create_health_food_database()
        self.feature_names = [
            'age', 'has_diabetes', 'has_hypertension', 'climate_hot',
            'stress_level', 'has_nut_allergy', 'budget_per_meal', 'weight_loss_goal'
        ]

    def create_health_food_database(self):
        """Create reference database of health-profiled foods."""
        foods = [
            {'food_name': 'Steamed Salmon with Asparagus', 'calories_per_100g': 180, 'glycemic_index': 15,
             'diabetes_friendly': 1, 'good_for_hot_climate': 0, 'stress_relief': 1, 'contains_nuts': 0, 'estimated_cost_per_serving': 6.50},
            {'food_name': 'Spinach & Berry Salad', 'calories_per_100g': 85, 'glycemic_index': 25,
             'diabetes_friendly': 1, 'good_for_hot_climate': 1, 'stress_relief': 1, 'contains_nuts': 0, 'estimated_cost_per_serving': 3.50},
            {'food_name': 'Walnut & Quinoa Bowl', 'calories_per_100g': 220, 'glycemic_index': 35,
             'diabetes_friendly': 1, 'good_for_hot_climate': 0, 'stress_relief': 1, 'contains_nuts': 1, 'estimated_cost_per_serving': 4.50},
            {'food_name': 'Cucumber Mint Yogurt Bowl', 'calories_per_100g': 60, 'glycemic_index': 20,
             'diabetes_friendly': 1, 'good_for_hot_climate': 1, 'stress_relief': 1, 'contains_nuts': 0, 'estimated_cost_per_serving': 2.00},
            {'food_name': 'Grilled Chicken & Steamed Broccoli', 'calories_per_100g': 140, 'glycemic_index': 10,
             'diabetes_friendly': 1, 'good_for_hot_climate': 0, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 4.00},
            {'food_name': 'Almond Butter Oatmeal', 'calories_per_100g': 210, 'glycemic_index': 40,
             'diabetes_friendly': 0, 'good_for_hot_climate': 0, 'stress_relief': 1, 'contains_nuts': 1, 'estimated_cost_per_serving': 3.00},
            {'food_name': 'Cold Tofu & Seaweed Salad', 'calories_per_100g': 75, 'glycemic_index': 15,
             'diabetes_friendly': 1, 'good_for_hot_climate': 1, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 2.50},
            {'food_name': 'Lentil & Vegetable Soup', 'calories_per_100g': 110, 'glycemic_index': 30,
             'diabetes_friendly': 1, 'good_for_hot_climate': 0, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 2.20},
            {'food_name': 'Honey Glazed Roasted Carrots', 'calories_per_100g': 120, 'glycemic_index': 65,
             'diabetes_friendly': 0, 'good_for_hot_climate': 0, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 1.80},
            {'food_name': 'Banana Honey Smoothie Bowl', 'calories_per_100g': 195, 'glycemic_index': 60,
             'diabetes_friendly': 0, 'good_for_hot_climate': 1, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 2.50},
            {'food_name': 'Brown Rice Risotto with Mushrooms', 'calories_per_100g': 160, 'glycemic_index': 50,
             'diabetes_friendly': 0, 'good_for_hot_climate': 0, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 3.20},
            {'food_name': 'Chia Seed & Dark Chocolate Pudding', 'calories_per_100g': 150, 'glycemic_index': 20,
             'diabetes_friendly': 1, 'good_for_hot_climate': 1, 'stress_relief': 1, 'contains_nuts': 0, 'estimated_cost_per_serving': 2.80},
            {'food_name': 'Kale, Apple & Ginger Detox Juice', 'calories_per_100g': 55, 'glycemic_index': 30,
             'diabetes_friendly': 1, 'good_for_hot_climate': 1, 'stress_relief': 1, 'contains_nuts': 0, 'estimated_cost_per_serving': 2.60},
            {'food_name': 'Avocado & Herb Poached Eggs', 'calories_per_100g': 175, 'glycemic_index': 15,
             'diabetes_friendly': 1, 'good_for_hot_climate': 0, 'stress_relief': 1, 'contains_nuts': 0, 'estimated_cost_per_serving': 3.80},
            {'food_name': 'Sweet Potato Mash with Coconut Oil', 'calories_per_100g': 145, 'glycemic_index': 55,
             'diabetes_friendly': 0, 'good_for_hot_climate': 0, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 1.90},
            {'food_name': 'Steamed Cod with Lemon & Herbs', 'calories_per_100g': 105, 'glycemic_index': 5,
             'diabetes_friendly': 1, 'good_for_hot_climate': 1, 'stress_relief': 0, 'contains_nuts': 0, 'estimated_cost_per_serving': 5.20}
        ]
        return pd.DataFrame(foods)

    def train_model(self):
        """Train RandomForest classifier on synthetic/profiled health data."""
        np.random.seed(self.random_state)
        n_samples = 300
        X = np.random.randn(n_samples, len(self.feature_names))
        # Binary target: 1 for highly recommended
        y = (X[:, 1] * 0.4 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        self.general_health_model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        self.general_health_model.fit(X, y)
        self.is_trained = True

    def get_feature_importance(self):
        """Return feature importance dictionary."""
        if not self.is_trained or self.general_health_model is None:
            self.train_model()
        importances = self.general_health_model.feature_importances_
        return {feat: float(imp) for feat, imp in zip(self.feature_names, importances)}

    def get_health_recommendations(self, user_profile, n_recommendations=5):
        """Generate ranked health recommendations filtered by allergies and conditions."""
        if self.general_health_model is None:
            raise ValueError("Model is not trained or has been corrupted.")

        # Extract features with safe fallback defaults
        feat_vector = []
        for feat in self.feature_names:
            val = user_profile.get(feat, 0)
            # Normalize extreme values
            if feat == 'age':
                val = max(18, min(100, val))
            elif feat == 'stress_level':
                val = max(1, min(10, val))
            elif feat == 'budget_per_meal':
                val = max(1.0, val)
            feat_vector.append(float(val))

        feat_vector = np.array(feat_vector).reshape(1, -1)
        base_prob = float(self.general_health_model.predict_proba(feat_vector)[0][1])

        df = self.food_db.copy()

        # Allergy filtering
        if user_profile.get('has_nut_allergy', 0) == 1:
            df = df[df['contains_nuts'] == 0]

        has_diabetes = user_profile.get('has_diabetes', 0)
        climate_hot = user_profile.get('climate_hot', 0)
        stress_level = user_profile.get('stress_level', 0)
        weight_loss = user_profile.get('weight_loss_goal', 0)

        # Score foods
        scored = []
        for idx, row in df.iterrows():
            prob = base_prob
            reasons = []

            # Tailor scoring to conditions
            if has_diabetes:
                if row['diabetes_friendly'] == 1:
                    prob += 0.30
                    reasons.append("Low glycemic index for blood sugar control")
                else:
                    prob -= 0.35
            else:
                # Non-diabetic baseline preference for standard nutrient-dense carbs
                if row['glycemic_index'] > 40:
                    prob += 0.10

            if climate_hot and row['good_for_hot_climate'] == 1:
                prob += 0.15
                reasons.append("Cooling and hydrating for hot climate")

            if stress_level >= 6:
                if row['stress_relief'] == 1:
                    prob += 0.25
                    reasons.append("Rich in magnesium and antioxidants for stress reduction")
                else:
                    prob -= 0.10

            if weight_loss:
                if row['calories_per_100g'] < 120:
                    prob += 0.20
                    reasons.append("Low calorie density for deficit")
                elif row['calories_per_100g'] > 180:
                    prob -= 0.15

            prob = float(max(0.01, min(0.99, prob)))
            reasoning = "; ".join(reasons) if reasons else "Nutritionally balanced meal option"

            scored.append({
                'food_name': row['food_name'],
                'probability': round(prob, 2),
                'reasoning': reasoning,
                'calories_per_100g': int(row['calories_per_100g']),
                'contains_nuts': int(row['contains_nuts'])
            })

        scored.sort(key=lambda x: x['probability'], reverse=True)
        return scored[:n_recommendations]
