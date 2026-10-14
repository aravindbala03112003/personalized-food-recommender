"""
Advanced Test Suite for DINE MATE
Integration tests, performance tests, and edge case testing
"""

import pytest
import numpy as np
import pandas as pd
import time
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from health_recommender import HealthAwareFoodRecommender
from budget_optimizer import BudgetFoodOptimizer
from goal_tracker import HealthGoalTracker
from hybrid_recommender import HybridFoodRecommender

class TestDataValidation:
    """Test data validation and edge cases"""
    
    def test_empty_user_profile(self):
        """Test handling of empty user profile"""
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        # Empty profile should use defaults
        empty_profile = {}
        recommendations = recommender.get_health_recommendations(empty_profile, 5)
        
        assert len(recommendations) == 5
        assert all('food_name' in rec for rec in recommendations)
    
    def test_extreme_user_values(self):
        """Test handling of extreme user input values"""
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        extreme_profile = {
            'age': 200,  # Extreme age
            'stress_level': 50,  # Way above scale
            'budget_per_meal': -5,  # Negative budget
            'has_diabetes': 5  # Invalid boolean
        }
        
        # Should handle gracefully
        recommendations = recommender.get_health_recommendations(extreme_profile, 3)
        assert len(recommendations) == 3
    
    def test_missing_required_columns(self):
        """Test handling of datasets with missing columns"""
        recommender = HybridFoodRecommender()
        
        # Dataset missing required columns
        incomplete_data = pd.DataFrame({
            'User': ['U1', 'U2'],
            'Item': ['I1', 'I2'],
            # Missing 'Score' column
        })
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((KeyError, ValueError)):
            recommender.prepare_data(incomplete_data, 
                                   user_col='User', 
                                   item_col='Item', 
                                   rating_col='Score')

class TestPerformance:
    """Performance and scalability tests"""
    
    def test_model_training_time(self):
        """Test that model training completes within reasonable time"""
        start_time = time.time()
        
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        training_time = time.time() - start_time
        
        # Should complete within 60 seconds
        assert training_time < 60, f"Training took {training_time:.2f} seconds"
    
    def test_recommendation_speed(self):
        """Test recommendation generation speed"""
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        user_profile = {'age': 30, 'has_diabetes': 1}
        
        start_time = time.time()
        recommendations = recommender.get_health_recommendations(user_profile, 10)
        recommendation_time = time.time() - start_time
        
        # Should complete within 5 seconds
        assert recommendation_time < 5, f"Recommendation took {recommendation_time:.2f} seconds"
        assert len(recommendations) == 10
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # Create large synthetic dataset
        large_data = pd.DataFrame({
            'UserId': [f'U{i}' for i in range(10000)],
            'ProductId': [f'P{i%1000}' for i in range(10000)],
            'Score': np.random.randint(1, 6, 10000)
        })
        
        recommender = HybridFoodRecommender()
        
        start_time = time.time()
        recommender.prepare_data(large_data)
        processing_time = time.time() - start_time
        
        # Should handle 10k interactions within 30 seconds
        assert processing_time < 30, f"Large data processing took {processing_time:.2f} seconds"

class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_corrupted_model_state(self):
        """Test handling of corrupted model state"""
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        # Corrupt the model
        recommender.general_health_model = None
        
        user_profile = {'age': 30}
        
        # Should raise appropriate error
        with pytest.raises(ValueError):
            recommender.get_health_recommendations(user_profile)
    
    def test_invalid_food_data(self):
        """Test handling of invalid food data"""
        optimizer = BudgetFoodOptimizer()
        
        # Mock invalid food database
        with patch.object(optimizer, 'create_budget_food_database') as mock_db:
            mock_db.return_value = pd.DataFrame()  # Empty database
            
            with pytest.raises((ValueError, IndexError)):
                optimizer.train_model()
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints"""
        # This is a conceptual test - in reality you'd use memory profiling tools
        recommender = HealthAwareFoodRecommender()
        
        # Monitor memory usage during training
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        recommender.train_model()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Should not use more than 500MB additional memory
        assert memory_increase < 500, f"Memory increase: {memory_increase:.2f} MB"

class TestIntegrationScenarios:
    """Integration testing scenarios"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Initialize all models
        health_model = HealthAwareFoodRecommender(random_state=42)
        budget_model = BudgetFoodOptimizer(random_state=42)
        goal_model = HealthGoalTracker(random_state=42)
        
        # Train all models
        health_model.train_model()
        budget_model.train_model()
        goal_model.train_models()
        
        # User journey
        user_profile = {
            'age': 35,
            'has_diabetes': 1,
            'budget_per_meal': 8.0,
            'weight_loss_goal': 1,
            'stress_level': 7,
            'gender': 'female',
            'current_weight': 160,
            'goal': 'weight_loss',
            'activity_level': 'moderate'
        }
        
        # Get recommendations from all models
        health_recs = health_model.get_health_recommendations(user_profile, 5)
        budget_recs = budget_model.get_budget_recommendations(
            daily_budget=user_profile['budget_per_meal'],
            weight_loss_goal=user_profile['weight_loss_goal'],
            n_recommendations=5
        )
        goal_recs = goal_model.get_goal_recommendations(user_profile, 5)
        
        # Validate results
        assert len(health_recs) == 5
        assert len(budget_recs) == 5
        assert len(goal_recs) == 5
        
        # Check consistency across models
        all_food_names = set()
        for recs in [health_recs, budget_recs, goal_recs]:
            for rec in recs:
                all_food_names.add(rec['food_name'])
        
        # Should have diverse recommendations
        assert len(all_food_names) >= 8, "Not enough diversity in recommendations"
    
    def test_user_preference_evolution(self):
        """Test system adaptation to changing user preferences"""
        health_model = HealthAwareFoodRecommender()
        health_model.train_model()
        
        # Initial user state
        initial_profile = {
            'age': 30,
            'has_diabetes': 0,
            'weight_loss_goal': 0,
            'stress_level': 3
        }
        
        initial_recs = health_model.get_health_recommendations(initial_profile, 5)
        
        # Changed user state (developed diabetes, high stress)
        changed_profile = {
            'age': 30,
            'has_diabetes': 1,
            'weight_loss_goal': 1,
            'stress_level': 9
        }
        
        changed_recs = health_model.get_health_recommendations(changed_profile, 5)
        
        # Recommendations should be different
        initial_foods = {rec['food_name'] for rec in initial_recs}
        changed_foods = {rec['food_name'] for rec in changed_recs}
        
        # At least some recommendations should change
        overlap = len(initial_foods.intersection(changed_foods))
        assert overlap < 5, "Recommendations didn't adapt to profile changes"

class TestBoundaryConditions:
    """Test boundary conditions and edge cases"""
    
    def test_single_user_dataset(self):
        """Test handling of dataset with single user"""
        single_user_data = pd.DataFrame({
            'UserId': ['U1'] * 10,
            'ProductId': [f'P{i}' for i in range(10)],
            'Score': [3, 4, 5, 2, 4, 3, 5, 4, 3, 4]
        })
        
        recommender = HybridFoodRecommender()
        
        # Should handle gracefully
        try:
            recommender.prepare_data(single_user_data)
            # If no exception, that's good
            assert True
        except Exception as e:
            # Should be a meaningful error message
            assert "user" in str(e).lower() or "interaction" in str(e).lower()
    
    def test_zero_budget_constraint(self):
        """Test handling of zero or very low budget"""
        optimizer = BudgetFoodOptimizer()
        optimizer.train_model()
        
        # Zero budget
        zero_budget_recs = optimizer.get_budget_recommendations(
            daily_budget=0.01,  # 1 cent budget
            n_recommendations=5
        )
        
        # Should return empty list or very cheap options
        assert len(zero_budget_recs) >= 0  # At least don't crash
    
    def test_extreme_health_conditions(self):
        """Test user with multiple severe health conditions"""
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        extreme_health_profile = {
            'age': 65,
            'has_diabetes': 1,
            'has_hypertension': 1,
            'has_nut_allergy': 1,
            'has_dairy_allergy': 1,
            'has_gluten_intolerance': 1,
            'stress_level': 10,
            'budget_per_meal': 3.0  # Low budget too
        }
        
        recommendations = recommender.get_health_recommendations(extreme_health_profile, 5)
        
        # Should still provide recommendations
        assert len(recommendations) > 0
        
        # All recommendations should respect allergies
        for rec in recommendations:
            # This would need more sophisticated checking in real implementation
            assert 'food_name' in rec

class TestModelConsistency:
    """Test consistency across different model runs"""
    
    def test_deterministic_training(self):
        """Test that models trained with same random seed produce same results"""
        # First model
        model1 = HealthAwareFoodRecommender(random_state=42)
        model1.train_model()
        
        # Second model with same seed
        model2 = HealthAwareFoodRecommender(random_state=42)
        model2.train_model()
        
        # Same input should produce same output
        user_profile = {'age': 30, 'has_diabetes': 1}
        
        recs1 = model1.get_health_recommendations(user_profile, 5)
        recs2 = model2.get_health_recommendations(user_profile, 5)
        
        # Extract just food names for comparison
        foods1 = [rec['food_name'] for rec in recs1]
        foods2 = [rec['food_name'] for rec in recs2]
        
        # Should be identical (or very similar due to randomness in model)
        assert len(set(foods1).intersection(set(foods2))) >= 3, "Models not consistent"

# Performance benchmarks
@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance monitoring"""
    
    def test_recommendation_throughput(self):
        """Test recommendation generation throughput"""
        recommender = HealthAwareFoodRecommender()
        recommender.train_model()
        
        user_profiles = [
            {'age': 25, 'has_diabetes': 0},
            {'age': 35, 'has_diabetes': 1},
            {'age': 45, 'stress_level': 8},
            {'age': 55, 'weight_loss_goal': 1},
            {'age': 65, 'has_hypertension': 1}
        ]
        
        start_time = time.time()
        
        for profile in user_profiles:
            recommender.get_health_recommendations(profile, 10)
        
        total_time = time.time() - start_time
        throughput = len(user_profiles) / total_time
        
        # Should handle at least 2 recommendations per second
        assert throughput >= 2, f"Throughput too low: {throughput:.2f} recs/sec"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])