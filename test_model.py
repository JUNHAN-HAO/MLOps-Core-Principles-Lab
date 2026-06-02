import os
import json

def test_winning_model_accuracy():
    """
    CI Test: Verifies that the winning model meets the production quality threshold.
    Requirement: Accuracy must be >= 0.85
    """
    metrics_file = "metrics.json"
    
    assert os.path.exists(metrics_file), f"CI Error: {metrics_file} was not found!"
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    winning_accuracy = metrics.get("accuracy")
    winning_model_name = metrics.get("algorithm", "Unknown Model")
    
    print(f"\n--- Running CI Pipeline Test ---")
    print(f"Loaded Winning Model: {winning_model_name}")
    print(f"Winning Model Accuracy: {winning_accuracy:.4f}")
    print(f"Target Threshold: 0.8500")
    
    assert winning_accuracy >= 0.85, (
        f"❌ CI Pipeline Failed! The winning model accuracy ({winning_accuracy:.4f}) "
        f"is below the required threshold of 0.85."
    )
    
    print("✅ CI Pipeline Passed! Model meets production quality requirements.\n")

if __name__ == \"__main__\":
    test_winning_model_accuracy()
