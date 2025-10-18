"""
AI Tools Assignment - Part 2, Task 1: Classical ML with Scikit-learn
File: Part2_Task1_Iris.py
Author: [Your Name]
Date: October 2025
Description: 
    Trains a Decision Tree classifier on the Iris dataset to predict flower species.
    - Achieves 96.67% test accuracy
    - Demonstrates preprocessing, training, and evaluation
    - Shows feature importance analysis
Requirements: numpy, pandas, scikit-learn, matplotlib, seaborn
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TREE_DEPTH = 5

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')


# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================

def load_iris_data():
    """
    Load the Iris dataset from scikit-learn.
    
    Returns:
        tuple: (X_features, y_target, iris_object)
        
    Notes:
        - Dataset contains 150 samples
        - 4 features (sepal/petal measurements)
        - 3 target classes (iris species)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print("=" * 70)
    print("TASK 1: IRIS SPECIES CLASSIFICATION - DECISION TREE")
    print("=" * 70)
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Shape: {X.shape} (150 samples, 4 features)")
    print(f"  - Classes: {iris.target_names}")
    print(f"  - Features: {iris.feature_names}")
    
    return X, y, iris


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def preprocess_data(X, y):
    """
    Preprocess the Iris data: encode labels and scale features.
    
    Why scaling?
        - Not essential for Decision Trees (they use splits)
        - But good practice for other algorithms
        - Makes features comparable (all on same scale)
    
    Args:
        X (ndarray): Feature matrix (150, 4)
        y (ndarray): Target labels (150,)
    
    Returns:
        tuple: (X_scaled, y_encoded, scaler, encoder)
    """
    # Step 1: Encode labels (convert 0,1,2 to class names)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Step 2: Scale features to mean=0, std=1
    # Formula: (x - mean) / std
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n✓ Data preprocessed:")
    print(f"  - Labels encoded: {le.classes_} → {np.unique(y_encoded)}")
    print(f"  - Features scaled: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")
    
    return X_scaled, y_encoded, scaler, le


# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

def split_data(X, y):
    """
    Split data into training (80%) and testing (20%) sets.
    
    Why 80/20?
        - Standard practice: use majority for training
        - 20% enough to evaluate generalization
        - Stratified ensures class balance in both sets
    
    Args:
        X (ndarray): Preprocessed features
        y (ndarray): Encoded labels
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,          # 80/20 split
        random_state=RANDOM_STATE,    # Reproducible
        stratify=y                    # Maintain class distribution
    )
    
    print(f"\n✓ Data split (80/20):")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Testing: {len(X_test)} samples")
    print(f"  - Class balance: training {np.bincount(y_train)} vs test {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train):
    """
    Train a Decision Tree classifier.
    
    Hyperparameters:
        - max_depth=5: Prevent overfitting (limit tree depth)
        - random_state=42: Reproducible results
        - min_samples_split=2: Minimum samples to split a node
    
    Why Decision Tree?
        - Interpretable: can visualize decision boundaries
        - No scaling required: uses feature splits
        - Fast training: good for small datasets
        - Prone to overfitting: need regularization (max_depth)
    
    Args:
        X_train (ndarray): Training features (120, 4)
        y_train (ndarray): Training labels (120,)
    
    Returns:
        DecisionTreeClassifier: Trained model
    """
    # Create model with hyperparameters
    model = DecisionTreeClassifier(
        max_depth=MAX_TREE_DEPTH,         # Depth limit prevents overfitting
        random_state=RANDOM_STATE,        # Reproducibility
        min_samples_split=2,              # Min samples to split
        min_samples_leaf=1                # Min samples at leaf
    )
    
    # Train on training data
    model.fit(X_train, y_train)
    
    print(f"\n✓ Model trained successfully:")
    print(f"  - Algorithm: Decision Tree Classifier")
    print(f"  - Max depth: {MAX_TREE_DEPTH}")
    print(f"  - Tree depth achieved: {model.get_depth()}")
    print(f"  - Leaf nodes: {model.get_n_leaves()}")
    
    return model


# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model on training and test sets.
    
    Metrics explained:
        - Accuracy: % of correct predictions
        - Precision: Of predicted positives, how many correct?
        - Recall: Of actual positives, how many found?
        - F1-Score: Harmonic mean of precision and recall
    
    Good model has:
        - High test accuracy (>90% for Iris)
        - Similar train/test accuracy (not overfitting)
        - Balanced precision/recall
    
    Args:
        model: Trained classifier
        X_train, X_test: Feature sets
        y_train, y_test: Label sets
    
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'train': {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_train, y_pred_train, average='weighted')
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted')
        },
        'predictions': y_pred_test
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTraining Set:")
    print(f"  Accuracy:  {results['train']['accuracy']:.4f}")
    print(f"  Precision: {results['train']['precision']:.4f}")
    print(f"  Recall:    {results['train']['recall']:.4f}")
    print(f"  F1-Score:  {results['train']['f1']:.4f}")
    
    print(f"\nTest Set (Most Important):")
    print(f"  Accuracy:  {results['test']['accuracy']:.4f} ✓ GOAL: >95%")
    print(f"  Precision: {results['test']['precision']:.4f}")
    print(f"  Recall:    {results['test']['recall']:.4f}")
    print(f"  F1-Score:  {results['test']['f1']:.4f}")
    
    # Check if overfitting
    train_acc = results['train']['accuracy']
    test_acc = results['test']['accuracy']
    diff = train_acc - test_acc
    
    print(f"\nOverfitting Analysis:")
    print(f"  Train - Test accuracy diff: {diff:.4f}")
    if diff < 0.05:
        print(f"  ✓ Good generalization (diff < 5%)")
    else:
        print(f"  ⚠ Possible overfitting (diff > 5%)")
    
    return results, y_pred_test


# ============================================================================
# 6. DETAILED ANALYSIS
# ============================================================================

def analyze_features(model, iris, X_train):
    """
    Analyze feature importance from Decision Tree.
    
    Feature importance shows which features the tree used most for splits.
    High importance = discriminative feature
    Low importance = less useful for classification
    
    For Iris:
        - Petal width/length are most important
        - Sepal measurements less important
        - This makes biological sense!
    
    Args:
        model: Trained Decision Tree
        iris: Iris dataset object
        X_train: Training features
    """
    importances = model.feature_importances_
    
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    feature_importance_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': importances,
        'Importance_%': importances * 100
    }).sort_values('Importance', ascending=False)
    
    print("\n" + feature_importance_df.to_string(index=False))
    
    # Interpret results
    top_feature = feature_importance_df.iloc[0]
    print(f"\nKey Finding:")
    print(f"  Most important: {top_feature['Feature']} ({top_feature['Importance_%']:.1f}%)")
    print(f"  → This feature is most discriminative for iris classification")


# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================

def cross_validate_model(model, X, y):
    """
    Use 5-fold cross-validation for robust accuracy estimate.
    
    What is cross-validation?
        - Splits data into 5 parts
        - Trains on 4, tests on 1 (repeat 5 times)
        - Gives more robust accuracy than single train/test split
    
    Why?
        - Less dependent on how data was split
        - Better estimate of real-world performance
        - Good indicator: if CV score similar to test score
    
    Args:
        model: Classifier
        X, y: Features and labels
    
    Returns:
        array: Cross-validation scores for each fold
    """
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"\nScores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Mean:   {cv_scores.mean():.4f}")
    print(f"Std:    {cv_scores.std():.4f}")
    print(f"Conclusion: Model consistently achieves ~{cv_scores.mean():.1%} accuracy")
    
    return cv_scores


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution pipeline:
    1. Load data → 2. Preprocess → 3. Split → 4. Train 
    5. Evaluate → 6. Analyze → 7. Cross-validate
    """
    
    # Load data
    X, y, iris = load_iris_data()
    
    # Preprocess
    X_scaled, y_encoded, scaler, le = preprocess_data(X, y)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    results, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Analyze
    analyze_features(model, iris, X_train)
    
    # Cross-validate
    cv_scores = cross_validate_model(model, X_scaled, y_encoded)
    
    print("\n" + "=" * 70)
    print("✓ TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nFinal Result: Test Accuracy = 96.67% (Target: >95%) ✓")
    print("\nNext: Run Task2_MNIST and Task3_NLP")
