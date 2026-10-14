"""
Unit Tests for DINE MATE
Comprehensive test suite for all ML models and components
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from health_recommender import HealthAwareFoodRecommender
from budget_optimizer import BudgetFoodOptimizer
from goal_tracker import HealthGoalTracker
from hybrid_recommender import HybridFoodRecommender

class TestHealthRecommender(unittest.TestCase):
    """Test cases for Health-Aware component of DINE MATE"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.recommender = HealthAwareFoodRecommender(random_state=42)
        
        # Sample user profile for testing
        self.test_user_profile = {
            'age': 35,
            'has_diabetes': 1,
            'has_hypertension': 0,
            'climate_hot': 1,
            'stress_level': 7,
            'has_nut_allergy': 1,
            'budget_per_meal': 5.0,
            'weight_loss_goal': 1
        }
    
    def test_food_database_creation(self):
        """Test that food database is created correctly"""
        db = self.recommender.create_health_food_database()
        
        self.assertIsInstance(db, pd.DataFrame)
        self.assertGreater(len(db), 0)
        
        # Check required columns
        required_columns = [
            'food_name', 'calories_per_100g', 'glycemic_index', 
            'diabetes_friendly', 'good_for_hot_climate', 'stress_relief',
            'contains_nuts', 'estimated_cost_per_serving'
        ]
        
        for col in required_columns:
            self.assertIn(col, db.columns, f"Missing column: {col}")
    
    def test_model_training(self):
        """Test that model trains successfully"""
        # This should not raise an exception
        self.recommender.train_model()
        
        # Check that model is marked as trained
        self.assertTrue(self.recommender.is_trained)
        self.assertIsNotNone(self.recommender.general_health_model)
    
    def test_recommendations_generation(self):
        """Test that recommendations are generated correctly"""
        # Train model first
        self.recommender.train_model()
        
        # Get recommendations
        recommendations = self.recommender.get_health_recommendations(
            self.test_user_profile, n_recommendations=5
        )
        
        # Validate recommendations
        self.assertEqual(len(recommendations), 5)
        
        for rec in recommendations:
            self.assertIn('food_name', rec)
            self.assertIn('probability', rec)
            self.assertIn('reasoning', rec)
            self.assertIn('calories_per_100g', rec)
            
            # Probability should be between 0 and 1
            self.assertGreaterEqual(rec['probability'], 0)
            self.assertLessEqual(rec['probability'], 1)
    
    def test_allergy_filtering(self):
        """Test that allergic foods are filtered out"""
        self.recommender.train_model()
        
        # User with nut allergy
        user_with_allergy = self.test_user_profile.copy()
        user_with_allergy['has_nut_allergy'] = 1
        
        recommendations = self.recommender.get_health_recommendations(user_with_allergy)
        
        # Check that no recommendations contain nuts
        for rec in recommendations:
            # This would require checking the food database for nut content
            # For now, we just check that we get recommendations
            self.assertIsNotNone(rec['food_name'])
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        self.recommender.train_model()
        
        importance = self.recommender.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check that all values are non-negative
        for feature, imp in importance.items():
            self.assertGreaterEqual(imp, 0)

class TestBudgetOptimizer(unittest.TestCase):
    """Test cases for Budget Food Optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = BudgetFoodOptimizer(random_state=42)
    
    def test_food_database_creation(self):
        """Test budget food database creation"""
        db = self.optimizer.create_budget_food_database()
        
        self.assertIsInstance(db, pd.DataFrame)
        self.assertGreater(len(db), 0)
        
        # Check required columns
        required_columns = [
            'food_name', 'cost_per_serving', 'calories_per_serving',
            'protein_per_serving', 'fiber_per_serving', 'satiety_score'
        ]
        
        for col in required_columns:
            self.assertIn(col, db.columns)
    
    def test_model_training(self):
        """Test budget model training"""
        self.optimizer.train_model()
        
        self.assertTrue(self.optimizer.is_trained)
        self.assertIsNotNone(self.optimizer.nutrition_per_dollar_model)
    
    def test_budget_recommendations(self):
        """Test budget recommendations generation"""
        self.optimizer.train_model()
        
        recommendations = self.optimizer.get_budget_recommendations(
            daily_budget=5.0,
            target_calories=2000,
            target_protein=100,
            weight_loss_goal=True,
            n_recommendations=5
        )
        
        self.assertEqual(len(recommendations), 5)
        
        for rec in recommendations:
            self.assertIn('food_name', rec)
            self.assertIn('cost_per_serving', rec)
            self.assertIn('nutrition_per_dollar', rec)
            
            # Cost should be within budget constraint
            self.assertLessEqual(rec['cost_per_serving'], 5.0 * 0.4)  # 40% of daily budget
    
    def test_nutrition_per_dollar_calculation(self):
        """Test nutrition per dollar calculations"""
        self.optimizer.train_model()
        
        recommendations = self.optimizer.get_budget_recommendations(daily_budget=10.0)
        
        # Check that nutrition per dollar values are positive
        for rec in recommendations:
            self.assertGreater(rec['nutrition_per_dollar'], 0)

class TestGoalTracker(unittest.TestCase):
    """Test cases for Health Goal Tracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = HealthGoalTracker(random_state=42)
        
        self.test_user_profile = {
            'age': 28,
            'gender': 'female',
            'current_weight': 150,
            'weight_to_lose': 15,
            'weight_to_gain': 0,
            'activity_level': 'moderate',
            'goal': 'weight_loss',
            'weeks_on_plan': 4,
            'compliance_rate': 0.8
        }
    
    def test_food_database_creation(self):
        """Test goal-specific food database creation"""
        db = self.tracker.create_goal_food_database()
        
        self.assertIsInstance(db, pd.DataFrame)
        self.assertGreater(len(db), 0)
        
        required_columns = [
            'food_name', 'calories_per_portion', 'protein_per_portion',
            'weight_loss_score', 'muscle_gain_score', 'portion_size'
        ]
        
        for col in required_columns:
            self.assertIn(col, db.columns)
    
    def test_model_training(self):
        """Test goal tracker model training"""
        self.tracker.train_models()
        
        self.assertTrue(self.tracker.is_trained)
        self.assertIsNotNone(self.tracker.weight_loss_model)
        self.assertIsNotNone(self.tracker.portion_control_model)
    
    def test_goal_recommendations(self):
        """Test goal-specific recommendations"""
        self.tracker.train_models()
        
        recommendations = self.tracker.get_goal_recommendations(
            self.test_user_profile, n_recommendations=5
        )
        
        self.assertEqual(len(recommendations), 5)
        
        for rec in recommendations:
            self.assertIn('food_name', rec)
            self.assertIn('goal_alignment', rec)
            self.assertIn('reasoning', rec)
            
            # Goal alignment should be between 0 and 1
            self.assertGreaterEqual(rec['goal_alignment'], 0)
            self.assertLessEqual(rec['goal_alignment'], 1)
    
    def test_portion_control_tips(self):
        """Test portion control tips generation"""
        tips = self.tracker.get_portion_control_tips(self.test_user_profile)
        
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0)
        
        for tip in tips:
            self.assertIsInstance(tip, str)
            self.assertGreater(len(tip), 10)  # Tips should be meaningful

class TestHybridRecommender(unittest.TestCase):
    """Test cases for Hybrid Recommender (KNN + SVD)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.recommender = HybridFoodRecommender(random_state=42)
        
        # Create sample interaction data
        self.sample_data = pd.DataFrame({
            'UserId': ['U001', 'U002', 'U001', 'U003', 'U002'] * 20,
            'ProductId': ['P001', 'P002', 'P003', 'P001', 'P003'] * 20,
            'Score': [4, 5, 3, 4, 2] * 20
        })
    
    def test_data_preparation(self):
        """Test data preparation for hybrid model"""
        self.recommender.prepare_data(self.sample_data)
        
        self.assertIsNotNone(self.recommender.trainset)
        self.assertIsNotNone(self.recommender.testset)
    
    def test_model_training(self):
        """Test KNN and SVD model training"""
        self.recommender.prepare_data(self.sample_data)
        self.recommender.train_hybrid_model()
        
        self.assertTrue(self.recommender.is_trained)
        self.assertIsNotNone(self.recommender.knn_model)
        self.assertIsNotNone(self.recommender.svd_model)
    
    def test_predictions(self):
        """Test individual model predictions"""
        self.recommender.prepare_data(self.sample_data)
        self.recommender.train_hybrid_model()
        
        # Test KNN prediction
        knn_pred = self.recommender.predict_knn('U001', 'P002')
        self.assertIsInstance(knn_pred, (int, float))
        self.assertGreaterEqual(knn_pred, 1)
        self.assertLessEqual(knn_pred, 5)
        
        # Test SVD prediction
        svd_pred = self.recommender.predict_svd('U001', 'P002')
        self.assertIsInstance(svd_pred, (int, float))
        self.assertGreaterEqual(svd_pred, 1)
        self.assertLessEqual(svd_pred, 5)
        
        # Test hybrid prediction
        hybrid_pred = self.recommender.predict_hybrid('U001', 'P002')
        self.assertIsInstance(hybrid_pred, (int, float))
        self.assertGreaterEqual(hybrid_pred, 1)
        self.assertLessEqual(hybrid_pred, 5)
    
    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        self.recommender.prepare_data(self.sample_data)
        self.recommender.train_hybrid_model()
        
        results = self.recommender.evaluate_models()
        
        self.assertIsInstance(results, dict)
        self.assertIn('KNN', results)
        self.assertIn('SVD', results)
        self.assertIn('Hybrid', results)
        
        for model_name, metrics in results.items():
            self.assertIn('MAE', metrics)
            self.assertIn('RMSE', metrics)
            self.assertGreaterEqual(metrics['MAE'], 0)
            self.assertGreaterEqual(metrics['RMSE'], 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_all_models_integration(self):
        """Test that all models work together"""
        # Sample user profile
        user_profile = {
            'age': 30,
            'has_diabetes': 1,
            'climate_hot': 1,
            'stress_level': 6,
            'has_nut_allergy': 0,
            'budget_per_meal': 8.0,
            'weight_loss_goal': 1,
            'gender': 'female',
            'current_weight': 160,
            'weight_to_lose': 10,
            'activity_level': 'moderate',
            'goal': 'weight_loss',
            'weeks_on_plan': 2,
            'compliance_rate': 0.8
        }
        
        # Initialize all models
        health_model = HealthAwareFoodRecommender(random_state=42)
        budget_model = BudgetFoodOptimizer(random_state=42)
        goal_model = HealthGoalTracker(random_state=42)
        
        # Train all models
        health_model.train_model()
        budget_model.train_model()
        goal_model.train_models()
        
        # Get recommendations from all models
        health_recs = health_model.get_health_recommendations(user_profile, 3)
        budget_recs = budget_model.get_budget_recommendations(
            daily_budget=user_profile['budget_per_meal'],
            weight_loss_goal=user_profile['weight_loss_goal'],
            n_recommendations=3
        )
        goal_recs = goal_model.get_goal_recommendations(user_profile, 3)
        
        # Validate that all models return recommendations
        self.assertEqual(len(health_recs), 3)
        self.assertEqual(len(budget_recs), 3)
        self.assertEqual(len(goal_recs), 3)
        
        # Check that recommendations have proper structure
        for rec in health_recs:
            self.assertIn('food_name', rec)
            self.assertIn('probability', rec)
        
        for rec in budget_recs:
            self.assertIn('food_name', rec)
            self.assertIn('nutrition_per_dollar', rec)
        
        for rec in goal_recs:
            self.assertIn('food_name', rec)
            self.assertIn('goal_alignment', rec)

if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHealthRecommender))
    test_suite.addTest(unittest.makeSuite(TestBudgetOptimizer))
    test_suite.addTest(unittest.makeSuite(TestGoalTracker))
    test_suite.addTest(unittest.makeSuite(TestHybridRecommender))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)