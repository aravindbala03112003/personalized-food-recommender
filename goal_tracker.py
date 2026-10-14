"""
Health Goal Tracker Module for DINE MATE
Provides fitness and body composition goal-aligned food recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

class HealthGoalTracker:
    """Tracks fitness goals (weight loss, muscle gain) and recommends aligned foods."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_trained = False
        self.weight_loss_model = None
        self.portion_control_model = None
        self.food_db = self.create_goal_food_database()

    def create_goal_food_database(self):
        """Create database of foods tagged with goal alignment metrics."""
        foods = [
            {'food_name': 'Grilled Chicken Breast', 'calories_per_portion': 165, 'protein_per_portion': 31,
             'weight_loss_score': 9.2, 'muscle_gain_score': 9.5, 'portion_size': '150g'},
            {'food_name': 'Steamed White Fish with Broccoli', 'calories_per_portion': 140, 'protein_per_portion': 26,
             'weight_loss_score': 9.6, 'muscle_gain_score': 8.5, 'portion_size': '200g'},
            {'food_name': 'Greek Yogurt with Whey & Berries', 'calories_per_portion': 210, 'protein_per_portion': 28,
             'weight_loss_score': 8.8, 'muscle_gain_score': 9.0, 'portion_size': '250g'},
            {'food_name': 'Egg White & Vegetable Omelet', 'calories_per_portion': 130, 'protein_per_portion': 20,
             'weight_loss_score': 9.5, 'muscle_gain_score': 8.2, 'portion_size': '180g'},
            {'food_name': 'Quinoa, Edamame & Avocado Bowl', 'calories_per_portion': 320, 'protein_per_portion': 18,
             'weight_loss_score': 7.5, 'muscle_gain_score': 8.0, 'portion_size': '300g'},
            {'food_name': 'Lean Ground Turkey Lettuce Wraps', 'calories_per_portion': 185, 'protein_per_portion': 24,
             'weight_loss_score': 9.0, 'muscle_gain_score': 8.7, 'portion_size': '200g'}
        ]
        return pd.DataFrame(foods)

    def train_models(self):
        """Train models for weight loss scoring and portion prediction."""
        db = self.create_goal_food_database()
        X = db[['calories_per_portion', 'protein_per_portion']]
        y_class = (db['weight_loss_score'] > 8.5).astype(int)

        self.weight_loss_model = DecisionTreeClassifier(random_state=self.random_state)
        self.weight_loss_model.fit(X, y_class)

        self.portion_control_model = LinearRegression()
        # Predict portion satiety from calories and protein
        y_portion = db['calories_per_portion'] / (db['protein_per_portion'] + 1)
        self.portion_control_model.fit(X, y_portion)

        self.is_trained = True

    def get_goal_recommendations(self, user_profile, n_recommendations=5):
        """Generate goal-specific recommendations with normalized alignment score."""
        if not self.is_trained:
            self.train_models()

        df = self.food_db.copy()
        goal = user_profile.get('goal', 'weight_loss')
        weight_loss = user_profile.get('weight_loss_goal', 1)

        recommendations = []
        for _, row in df.iterrows():
            if weight_loss:
                raw_score = row['weight_loss_score']
                reason = "High satiety, lean protein density promotes calorie deficit"
            else:
                raw_score = row['muscle_gain_score']
                reason = "Optimal protein bioavailability supports hypertrophy and recovery"

            # Normalize to 0..1 scale
            alignment = float(raw_score / 10.0)
            alignment = max(0.0, min(1.0, alignment))

            recommendations.append({
                'food_name': row['food_name'],
                'goal_alignment': round(alignment, 2),
                'reasoning': reason
            })

        recommendations.sort(key=lambda x: x['goal_alignment'], reverse=True)
        return recommendations[:n_recommendations]

    def get_portion_control_tips(self, user_profile):
        """Return actionable, evidence-based portion control tips."""
        return [
            "Use the visual plate method: fill half your plate with leafy greens, one quarter with lean protein, and one quarter with complex carbs.",
            "Drink a glass of water 15 minutes before your meal to assist digestion and support accurate satiety signals.",
            "Eat protein and fibrous vegetables first to slow gastric emptying and stabilize postprandial glucose curves.",
            "Avoid eating directly from food packages; portion your meal onto a separate plate to prevent passive overconsumption."
        ]
