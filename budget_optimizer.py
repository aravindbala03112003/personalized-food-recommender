"""
Budget Food Optimizer Module for DINE MATE
Maximizes nutritional value per dollar spent within user budget constraints.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class BudgetFoodOptimizer:
    """Optimizes meal recommendations for maximum nutrition under budget constraints."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_trained = False
        self.nutrition_per_dollar_model = None
        self.food_db = self.create_budget_food_database()

    def create_budget_food_database(self):
        """Create database of budget-friendly foods with nutritional density."""
        foods = [
            {'food_name': 'Lentils & Brown Rice Bowl', 'cost_per_serving': 1.20, 'calories_per_serving': 380,
             'protein_per_serving': 18, 'fiber_per_serving': 14, 'satiety_score': 8.5},
            {'food_name': 'Scrambled Eggs with Spinach', 'cost_per_serving': 1.50, 'calories_per_serving': 240,
             'protein_per_serving': 16, 'fiber_per_serving': 3, 'satiety_score': 8.0},
            {'food_name': 'Chickpea Veggie Curry', 'cost_per_serving': 1.80, 'calories_per_serving': 350,
             'protein_per_serving': 14, 'fiber_per_serving': 10, 'satiety_score': 8.2},
            {'food_name': 'Peanut Butter Oatmeal', 'cost_per_serving': 0.85, 'calories_per_serving': 320,
             'protein_per_serving': 12, 'fiber_per_serving': 7, 'satiety_score': 8.7},
            {'food_name': 'Canned Tuna & White Bean Salad', 'cost_per_serving': 1.95, 'calories_per_serving': 310,
             'protein_per_serving': 28, 'fiber_per_serving': 8, 'satiety_score': 9.0},
            {'food_name': 'Tofu Stir Fry with Mixed Vegetables', 'cost_per_serving': 2.10, 'calories_per_serving': 280,
             'protein_per_serving': 20, 'fiber_per_serving': 6, 'satiety_score': 7.8},
            {'food_name': 'Greek Yogurt with Frozen Berries', 'cost_per_serving': 1.75, 'calories_per_serving': 190,
             'protein_per_serving': 17, 'fiber_per_serving': 4, 'satiety_score': 7.5},
            {'food_name': 'Black Bean Quesadilla', 'cost_per_serving': 1.40, 'calories_per_serving': 360,
             'protein_per_serving': 15, 'fiber_per_serving': 9, 'satiety_score': 8.1}
        ]
        return pd.DataFrame(foods)

    def train_model(self):
        """Train linear model predicting nutrition per dollar from cost and satiety."""
        db = self.create_budget_food_database()
        if db.empty:
            raise ValueError("Food database cannot be empty.")

        X = db[['cost_per_serving', 'calories_per_serving', 'protein_per_serving', 'fiber_per_serving']]
        # Nutrition density target
        y = (db['protein_per_serving'] * 2 + db['fiber_per_serving'] * 1.5 + db['satiety_score'] * 5) / db['cost_per_serving']

        self.nutrition_per_dollar_model = LinearRegression()
        self.nutrition_per_dollar_model.fit(X, y)
        self.is_trained = True

    def get_budget_recommendations(self, daily_budget=10.0, target_calories=2000, target_protein=100, weight_loss_goal=False, n_recommendations=5):
        """Get budget-optimized meal suggestions satisfying cost constraints."""
        if not self.is_trained:
            self.train_model()

        df = self.food_db.copy()

        # Hard constraint: cost per serving <= 40% of daily budget
        max_cost = float(daily_budget) * 0.4
        filtered = df[df['cost_per_serving'] <= max_cost].copy()

        if filtered.empty:
            # Fallback to cheapest items if strict budget
            filtered = df.nsmallest(n_recommendations, 'cost_per_serving').copy()

        X = filtered[['cost_per_serving', 'calories_per_serving', 'protein_per_serving', 'fiber_per_serving']]
        pred_scores = self.nutrition_per_dollar_model.predict(X)
        filtered['nutrition_per_dollar'] = [max(1.0, float(s)) for s in pred_scores]

        filtered = filtered.sort_values(by='nutrition_per_dollar', ascending=False)
        results = []
        for _, row in filtered.head(n_recommendations).iterrows():
            results.append({
                'food_name': row['food_name'],
                'cost_per_serving': float(row['cost_per_serving']),
                'nutrition_per_dollar': round(float(row['nutrition_per_dollar']), 2)
            })
        return results
