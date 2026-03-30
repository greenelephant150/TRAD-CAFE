# save as create_sample_model.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Create sample training data
np.random.seed(42)
X_train = np.random.rand(100, 20)  # 100 samples, 20 features
y_train = np.random.randint(0, 2, 100)

# Train a simple model
model = RandomForestClassifier(n_estimators=10, max_depth=5)
model.fit(X_train, y_train)

# Save model
os.makedirs('ai/trained_models', exist_ok=True)
with open('ai/trained_models/random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Sample model created at ai/trained_models/random_forest.pkl")