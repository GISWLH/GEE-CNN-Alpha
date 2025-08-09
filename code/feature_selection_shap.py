import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Arial font for matplotlib
plt.rcParams['font.family'] = 'Arial'

print("Starting feature selection analysis...")

# Read the CSV data
print("Loading data...")
data = pd.read_csv('data/alphaearth_extracted_values.csv')
print(f"Data shape: {data.shape}")

# Extract Alpha features (Alpha1 to Alpha64)
alpha_columns = [col for col in data.columns if col.startswith('Alpha')]
print(f"Found {len(alpha_columns)} Alpha features: {alpha_columns[:5]}...")

X = data[alpha_columns]
y = data['landcover']

print(f"Features shape: {X.shape}")
print(f"Target classes: {np.unique(y)}")
print(f"Class distribution:")
print(y.value_counts())

# Remove any rows with NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]
print(f"After removing NaN values: {X.shape[0]} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create and train Random Forest model
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': alpha_columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 features by Random Forest importance:")
print(feature_importance.head(10))

# SHAP analysis
print("\nPerforming SHAP analysis...")
explainer = shap.TreeExplainer(rf)

# Use a subset for SHAP calculation to speed up
sample_size = min(1000, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# For multi-class, get mean absolute SHAP values across all classes
if len(shap_values) > 1:
    mean_shap_values = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
else:
    mean_shap_values = np.abs(shap_values[0]).mean(0)

# Create SHAP importance dataframe
shap_importance = pd.DataFrame({
    'feature': alpha_columns,
    'shap_importance': mean_shap_values
}).sort_values('shap_importance', ascending=False)

print("\nTop 10 features by SHAP importance:")
print(shap_importance.head(10))

# Get top 5 most important Alpha features
top_5_features = shap_importance.head(5)
print("\n" + "="*50)
print("TOP 5 MOST IMPORTANT ALPHA FEATURES:")
print("="*50)
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    print(f"{i}. {row['feature']}: SHAP importance = {row['shap_importance']:.6f}")
print("="*50)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Importance Analysis for AlphaEarth Bands', fontsize=16, fontweight='bold')

# Plot 1: Random Forest Feature Importance (Top 15)
axes[0, 0].barh(range(15), feature_importance.head(15)['importance'][::-1])
axes[0, 0].set_yticks(range(15))
axes[0, 0].set_yticklabels(feature_importance.head(15)['feature'][::-1])
axes[0, 0].set_xlabel('Importance Score')
axes[0, 0].set_title('Random Forest Feature Importance (Top 15)')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: SHAP Feature Importance (Top 15)
axes[0, 1].barh(range(15), shap_importance.head(15)['shap_importance'][::-1])
axes[0, 1].set_yticks(range(15))
axes[0, 1].set_yticklabels(shap_importance.head(15)['feature'][::-1])
axes[0, 1].set_xlabel('SHAP Importance Score')
axes[0, 1].set_title('SHAP Feature Importance (Top 15)')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: Top 5 features comparison
top_5_names = top_5_features['feature'].values
rf_scores = [feature_importance[feature_importance['feature']==f]['importance'].iloc[0] for f in top_5_names]
shap_scores = top_5_features['shap_importance'].values

x = np.arange(len(top_5_names))
width = 0.35

axes[1, 0].bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8)
axes[1, 0].bar(x + width/2, shap_scores, width, label='SHAP', alpha=0.8)
axes[1, 0].set_xlabel('Alpha Features')
axes[1, 0].set_ylabel('Importance Score')
axes[1, 0].set_title('Top 5 Features: RF vs SHAP Importance')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(top_5_names, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Class distribution
class_counts = y.value_counts()
axes[1, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Target Class Distribution')

plt.tight_layout()

# Create image directory if it doesn't exist
import os
os.makedirs('image', exist_ok=True)

plt.savefig('image/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
results = {
    'top_5_alpha_features': top_5_features['feature'].tolist(),
    'shap_importance_scores': top_5_features['shap_importance'].tolist(),
    'rf_importance_scores': [feature_importance[feature_importance['feature']==f]['importance'].iloc[0] for f in top_5_features['feature']],
    'model_accuracy': accuracy,
    'total_features': len(alpha_columns),
    'sample_size': len(X)
}

# Save to CSV
top_5_features.to_csv('data/top_5_alpha_features.csv', index=False)
shap_importance.to_csv('data/all_alpha_shap_importance.csv', index=False)

print(f"\nAnalysis completed!")
print(f"Results saved to:")
print(f"- data/top_5_alpha_features.csv")
print(f"- data/all_alpha_shap_importance.csv") 
print(f"- image/feature_importance_analysis.png")
print(f"\nModel accuracy: {accuracy:.4f}")
print(f"Total samples processed: {len(X):,}")