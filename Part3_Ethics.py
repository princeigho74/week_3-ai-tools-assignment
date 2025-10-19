"""
AI Tools Assignment - Part 3: Ethics & Optimization
Bias Analysis, Mitigation Strategies, and Debugging Challenges

Author: AI Assignment
Date: 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PART 3: ETHICS & OPTIMIZATION - BIAS ANALYSIS AND DEBUGGING")
print("=" * 80)

# ============================================================================
# Section 1: Ethical Considerations - Bias Analysis in MNIST
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 1: ETHICAL CONSIDERATIONS - BIAS ANALYSIS IN MNIST")
print("=" * 80)

print("""
IDENTIFIED POTENTIAL BIASES IN MNIST CLASSIFIER:

1. DISTRIBUTION BIAS
   ├─ Issue: MNIST dataset may have class imbalance where some digits (0-9)
   │         are underrepresented compared to others
   ├─ Impact: Model performs worse on underrepresented digits
   └─ Scenario: If digit '8' has fewer samples, it gets lower accuracy

2. REPRESENTATION BIAS
   ├─ Issue: MNIST contains 28x28 handwritten digits from limited sources
   │         (mostly US postal service data)
   ├─ Impact: Model may not generalize to handwriting styles from other
   │         countries or age groups
   └─ Scenario: Elderly handwriting or non-Latin scripts show degraded performance

3. ANNOTATION BIAS
   ├─ Issue: If digit labels were determined by human annotators, subjective
   │         interpretations could introduce errors
   ├─ Impact: Model learns from potentially mislabeled examples
   └─ Scenario: Ambiguous digits (like '0' vs 'O') may be labeled inconsistently

4. GENDER AND DEMOGRAPHIC BIAS
   ├─ Issue: Training on specific demographic groups' handwriting
   ├─ Impact: Performance varies across demographic characteristics
   └─ Scenario: Model performs differently for handwriting of different age groups

5. HISTORICAL BIAS
   ├─ Issue: MNIST is from the 1990s; handwriting habits have changed
   ├─ Impact: Model may not recognize modern handwriting styles
   └─ Scenario: Digital pen inputs or modern writing styles underperform

═════════════════════════════════════════════════════════════════════════════════

MITIGATION STRATEGIES:

1. DATA-LEVEL MITIGATIONS
   ✓ Class Balancing: Use stratified sampling, oversampling, or SMOTE to ensure
     equal representation of all digit classes (0-9)
   ✓ Data Augmentation: Apply rotations, scaling, and elastic deformations to
     create diverse training examples
   ✓ Diversify Sources: Train on multiple handwriting datasets (EMNIST, IAM,
     Chars74K) from different countries and demographics
   ✓ Fairness Datasets: Use balanced datasets like "Balanced MNIST" with equal
     samples per class

2. ALGORITHMIC MITIGATIONS
   ✓ Fairness Constraints: Use TensorFlow Fairness Indicators to monitor
     performance across subgroups
   ✓ Threshold Adjustment: Calibrate decision thresholds differently for
     underrepresented classes
   ✓ Ensemble Methods: Combine multiple models trained on different subsets
     to reduce bias from any single model
   ✓ Adversarial Debiasing: Train a debiasing network to remove biased features

3. MONITORING AND EVALUATION
   ✓ Per-Class Metrics: Report accuracy, precision, recall separately for
     each digit (0-9)
   ✓ Fairness Audits: Conduct regular bias audits across demographic groups
   ✓ Error Analysis: Investigate which digit classes have highest error rates
   ✓ User Testing: Test on real-world data from diverse populations

4. SPACY NLP BIAS MITIGATION (Amazon Reviews)
   ├─ Language Bias: Different languages/accents may be misclassified
   ├─ Cultural Bias: Sentiment may vary by culture (e.g., "cheap" is negative
   │                 in developed countries but positive in developing ones)
   ├─ Review Source Bias: Professional vs consumer reviews have different patterns
   ├─ Mitigation: Use multilingual models, diversify training data, audit on
   │             different demographic review sources

5. PRACTICAL IMPLEMENTATION WITH TENSORFLOW FAIRNESS
""")

# Load MNIST and demonstrate fairness analysis
print("\n[Fairness Analysis] Loading MNIST and analyzing class distribution...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Analyze class distribution
train_distribution = np.bincount(y_train)
test_distribution = np.bincount(y_test)

print("\nClass Distribution Analysis:")
print("=" * 80)
print(f"{'Digit':<10} {'Train Count':<20} {'Train %':<15} {'Test Count':<15} {'Test %':<15}")
print("-" * 80)

for digit in range(10):
    train_pct = (train_distribution[digit] / len(y_train)) * 100
    test_pct = (test_distribution[digit] / len(y_test)) * 100
    print(f"{digit:<10} {train_distribution[digit]:<20} {train_pct:<15.2f} "
          f"{test_distribution[digit]:<15} {test_pct:<15.2f}")

print("\nBias Observation:")
min_class = np.argmin(train_distribution)
max_class = np.argmax(train_distribution)
imbalance_ratio = train_distribution[max_class] / train_distribution[min_class]
print(f"  Digit with most samples: {max_class} ({train_distribution[max_class]} samples)")
print(f"  Digit with least samples: {min_class} ({train_distribution[min_class]} samples)")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
print(f"  ⚠ Risk: Model may have bias towards digit {max_class}")

# ============================================================================
# Section 2: Troubleshooting Challenge - Buggy Code and Fixes
# ============================================================================

print("\n\n" + "=" * 80)
print("SECTION 2: TROUBLESHOOTING CHALLENGE - BUGGY CODE AND FIXES")
print("=" * 80)

print("""
BUGGY CODE PROVIDED:
═════════════════════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models

# BUGGY CODE - Do NOT use this
buggy_model = models.Sequential([
    layers.Dense(128, input_shape=(28, 28)),  # BUG 1: Wrong input shape
    layers.Activation('relu'),
    layers.Dense(64),
    layers.Dense(10, activation='sigmoid')    # BUG 2: Wrong activation function
])

buggy_model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',  # BUG 3: Loss-target mismatch
    metrics=['accuracy']
)

# BUGGY: Trying to fit with 2D data directly
buggy_model.fit(X_train, y_train, epochs=5)  # BUG 4: Dimension mismatch

═════════════════════════════════════════════════════════════════════════════════

IDENTIFIED BUGS AND EXPLANATIONS:

BUG 1: Wrong Input Shape
  ├─ Code: layers.Dense(128, input_shape=(28, 28))
  ├─ Issue: Dense layer expects flat input, but images are 28x28 (2D)
  ├─ Error: "ValueError: Input 0 of layer dense is incompatible with the layer"
  └─ Fix: Flatten images first or use input_shape=(784,) after flattening

BUG 2: Wrong Activation Function
  ├─ Code: layers.Dense(10, activation='sigmoid')
  ├─ Issue: For multi-class classification (10 digits), sigmoid is incorrect
  │        Sigmoid is for binary classification. For 10 classes, use softmax
  ├─ Error: Probabilities won't sum to 1, causing convergence issues
  └─ Fix: Use activation='softmax' for multi-class problems

BUG 3: Loss-Target Mismatch
  ├─ Code: loss='sparse_categorical_crossentropy' with one-hot encoded targets
  ├─ Issue: If labels are one-hot encoded (shape: (N, 10)), need
  │        'categorical_crossentropy' not 'sparse_categorical_crossentropy'
  ├─ Error: Shape mismatch or incorrect loss calculation
  └─ Fix: Match loss function to label format:
           - One-hot encoded → 'categorical_crossentropy'
           - Integer labels → 'sparse_categorical_crossentropy'

BUG 4: Dimension Mismatch
  ├─ Code: model.fit(X_train, y_train, ...) with X_train shape (60000, 28, 28)
  ├─ Issue: Model expects input shape (784,) but receives (28, 28)
  ├─ Error: "ValueError: Input 0 of layer is incompatible with the layer"
  └─ Fix: Reshape/Flatten data: X_train_flat = X_train.reshape(-1, 784)

═════════════════════════════════════════════════════════════════════════════════

CORRECTED CODE:
""")

# CORRECTED VERSION
print("\n[Debugging] Building CORRECTED Model...\n")

# Prepare data correctly
X_train_mnist, y_train_mnist = tf.keras.datasets.mnist.load_data()[0]
X_test_mnist, y_test_mnist = tf.keras.datasets.mnist.load_data()[1]

# FIX 4: Normalize and reshape data properly
X_train_flat = X_train_mnist.astype('float32').reshape(-1, 784) / 255.0
X_test_flat = X_test_mnist.astype('float32').reshape(-1, 784) / 255.0

# One-hot encode labels
y_train_cat = tf.keras.utils.to_categorical(y_train_mnist, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test_mnist, 10)

print("✓ Data prepared correctly:")
print(f"  - X_train shape: {X_train_flat.shape} (flattened from 28x28)")
print(f"  - y_train shape: {y_train_cat.shape} (one-hot encoded)")

# CORRECTED MODEL
corrected_model = models.Sequential([
    layers.Input(shape=(784,)),  # FIX 1: Correct input shape for flattened images
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # FIX 2: Softmax for multi-class
])

# FIX 3: Correct loss function for one-hot encoded labels
corrected_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Matches one-hot encoded labels
    metrics=['accuracy']
)

print("\n✓ Model compiled with correct configuration:")
print("  - Loss: categorical_crossentropy (matches one-hot labels)")
print("  - Output activation: softmax (for 10 classes)")
print("  - Input shape: (784,) (flattened 28x28 images)")

# Train corrected model
print("\n✓ Training corrected model...")
history_corrected = corrected_model.fit(
    X_train_flat, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)

# Evaluate
test_loss, test_acc = corrected_model.evaluate(X_test_flat, y_test_cat, verbose=0)
print(f"\n✓ Corrected Model Results:")
print(f"  - Test Accuracy: {test_acc:.4f}")
print(f"  - Test Loss: {test_loss:.4f}")

# ============================================================================
# Section 3: Model Optimization Techniques
# ============================================================================

print("\n\n" + "=" * 80)
print("SECTION 3: MODEL OPTIMIZATION TECHNIQUES")
print("=" * 80)

optimization_tips = """
1. HYPERPARAMETER OPTIMIZATION
   ├─ Learning Rate: Start with 0.001, adjust based on loss curve
   ├─ Batch Size: 32-128 for most cases; larger batches = faster but less stable
   ├─ Epochs: Monitor validation loss; stop when it stops improving (Early Stopping)
   └─ Example: Use ReduceLROnPlateau to decrease learning rate when stuck

2. MODEL ARCHITECTURE OPTIMIZATION
   ├─ Layer Sizing: Balance capacity (avoid underfitting) with complexity
   ├─ Dropout: Use 0.2-0.5 dropout to prevent overfitting
   ├─ Batch Normalization: Normalizes layer inputs for faster convergence
   ├─ Activation Functions: ReLU for hidden, softmax for multi-class output
   └─ Example: Add BatchNormalization after dense layers

3. REGULARIZATION TECHNIQUES
   ├─ L1/L2 Regularization: Penalizes large weights to prevent overfitting
   ├─ Early Stopping: Stop training when validation loss increases
   ├─ Data Augmentation: Creates more training samples from existing data
   └─ Example: callbacks=[EarlyStopping(monitor='val_loss', patience=3)]

4. TRAINING OPTIMIZATION
   ├─ Optimizer Choice: Adam (adaptive), SGD (stable), RMSprop (good default)
   ├─ Mixed Precision: Use float16 for faster computation on GPUs
   ├─ Gradient Clipping: Prevents exploding gradients
   └─ Example: optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

5. INFERENCE OPTIMIZATION
   ├─ Model Quantization: Reduce model size (fp32 → int8)
   ├─ Model Pruning: Remove non-critical weights
   ├─ TensorFlow Lite: Deploy on mobile/edge devices
   └─ Example: Use tf.lite.TFLiteConverter for mobile deployment

APPLIED OPTIMIZATION IN CORRECTED MODEL:
  ✓ Dropout layers: Reduces overfitting
  ✓ Moderate layer sizes: 128→64→10 (efficient)
  ✓ Adam optimizer: Adaptive learning rates
  ✓ Early stopping ready: Monitor validation metrics
"""

print(optimization_tips)

# ============================================================================
# Section 4: Fairness Report Summary
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: ETHICAL AI FAIRNESS REPORT SUMMARY")
print("=" * 80)

fairness_report = """
FAIRNESS AUDIT CHECKLIST:

✓ Data Bias Assessment
  └─ Analyzed class distribution in training/test sets
  └─ Identified potential representation gaps
  └─ Recommend: Use balanced sampling or data augmentation

✓ Model Performance Disparity
  └─ Monitor per-class accuracy across all 10 digits
  └─ Identify underperforming subgroups
  └─ Recommend: Weight loss function to focus on low-performing classes

✓ Transparency & Explainability
  └─ Document model architecture and training process
  └─ Provide feature importance insights
  └─ Recommend: Use LIME or SHAP for local explanations

✓ Stakeholder Impact Analysis
  └─ Consider real-world deployment context
  └─ Identify who benefits and who might be harmed
  └─ Recommend: Bias testing with diverse human samples

✓ Continuous Monitoring
  └─ Track performance drift over time
  └─ Monitor for emerging biases in production
  └─ Recommend: Establish feedback loops for ongoing audits

RECOMMENDATIONS FOR DEPLOYMENT:

1. Before Deployment
   ├─ Test on diverse handwriting styles and demographics
   ├─ Document all known limitations and biases
   ├─ Obtain stakeholder consent with full transparency
   └─ Create audit trail for accountability

2. During Deployment
   ├─ Monitor per-class performance metrics continuously
   ├─ Set up alerts for performance degradation
   ├─ Collect user feedback and error cases
   └─ Maintain version control of model updates

3. After Deployment
   ├─ Regular fairness audits (quarterly/annually)
   ├─ Retrain with new diverse data sources
   ├─ Update mitigation strategies based on findings
   └─ Engage with affected communities for feedback

ETHICS CONCLUSION:

AI systems are tools that reflect the biases in their training data and design
choices. By systematically identifying, measuring, and mitigating biases, we can
build more fair and trustworthy AI systems. This requires ongoing effort, diverse
perspectives, and commitment to ethical principles.
"""

print(fairness_report)

print("\n" + "=" * 80)
print("PART 3: ETHICS & OPTIMIZATION COMPLETED SUCCESSFULLY!")
print("=" * 80)
