import os
import numpy as np
import pandas as pd
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

directory = 'C:/Users/pc/Downloads/genres_original'

# Function to load and extract features from .wav files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs, Chroma, and Spectral Contrast
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
    return np.hstack([mfccs, chroma, spectral_contrast])

# Prepare dataset
def prepare_dataset(directory):
    genres = []
    features = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                genre = os.path.basename(root)
                feat = extract_features(file_path)
                genres.append(genre)
                features.append(feat)
    
    X = np.array(features)
    y = np.array(genres)
    return X, y

# Load and prepare data
X, y = prepare_dataset(directory)

# Standardize features for SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM model
svm_model = SVC()

# Set up hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best parameters and model
print("Best Parameters from Grid Search:", grid_search.best_params_)
best_svm = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_svm.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
