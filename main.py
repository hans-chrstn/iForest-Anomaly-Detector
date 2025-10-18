import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import shap

RANDOM_STATE = 42

DATA_FILE_PATH = "data/Industrial_fault_detection.csv"
OUTPUT_FOLDER = "outputs"

def setup_environment():
    print("--- Setting up the environment ---")
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory at: {OUTPUT_FOLDER}")

def load_and_preprocess_data(file_path):
    print(f"\n--- Loading and preprocessing data from: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The data file was not found at '{file_path}'.")
        print("Please make sure the CSV file is in the 'data' subfolder.")
        return None, None

    if 'Target' in df.columns:
        features = df.drop('Target', axis=1)
    else:
        features = df

    feature_names = features.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    print("Data loaded and scaled successfully.")
    print(f"Dataset shape: {X_scaled_df.shape}")
    return X_scaled_df, feature_names

def optimize_hyperparameters(X_scaled_df):
    print("\n--- SOP 2: Finding Optimal Model Hyperparameters ---")
    
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_samples': ['auto', 0.6, 0.75, 0.9],
        'contamination': [0.01, 0.02, 0.05, 0.1],
        'max_features': [0.5, 0.75, 1.0],
    }

    def dummy_scorer(estimator, X):
        return -np.mean(estimator.decision_function(X))

    iso_forest = IsolationForest(random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        iso_forest,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring=dummy_scorer,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    print("Running RandomizedSearchCV... (this may take a moment)")
    random_search.fit(X_scaled_df)

    print("Hyperparameter optimization complete.")
    print(f"Best Parameters Found: {random_search.best_params_}")
    return random_search.best_params_

def train_final_model(X_scaled_df, best_params):
    print("\n--- Training final model with optimal parameters ---")
    
    final_params = best_params.copy()
    final_params['random_state'] = RANDOM_STATE

    final_model = IsolationForest(**final_params)
    final_model.fit(X_scaled_df)
    
    anomaly_predictions = final_model.predict(X_scaled_df)
    
    print("Final model trained successfully.")
    return final_model, anomaly_predictions

def plot_anomalies_2d(X_scaled_df, predictions, feature_names):
    print("\n--- SOP 3 (Part 1): Generating 2D Anomaly Visualization via PCA ---")
    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled_df)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='coolwarm', s=20, alpha=0.7
    )
    plt.title('Figure 5.1: 2D Visualization of Detected Anomalies using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    handles, _ = scatter.legend_elements()
    legend_labels = ['Normal', 'Anomaly']
    plt.legend(handles, legend_labels, title="Status")
    
    save_path = os.path.join(OUTPUT_FOLDER, "PCA_Anomaly_Visualization.png")
    plt.savefig(save_path)
    plt.show()
    
    print(f"PCA plot saved to: {save_path}")

def generate_shap_plots(model, X_scaled_df, feature_names):
    print("\n--- SOP 3 (Part 2): Explaining Model Decisions with SHAP (The 'Why') ---")
    
    explainer = shap.TreeExplainer(model)
    
    print("Calculating SHAP values... (this can be slow on large datasets)")
    shap_values = explainer.shap_values(X_scaled_df)
    
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled_df, feature_names=feature_names, show=False)
    plt.title('Figure 5.2: SHAP Feature Importance for Anomaly Detection')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_FOLDER, "SHAP_Summary_Plot.png")
    plt.savefig(save_path)
    plt.show()
    
    print(f"SHAP summary plot saved to: {save_path}")

def main():
    setup_environment()
    
    X_scaled_df, feature_names = load_and_preprocess_data(DATA_FILE_PATH)
    
    if X_scaled_df is None:
        return

    best_params = optimize_hyperparameters(X_scaled_df)
    

    final_model, anomaly_predictions = train_final_model(X_scaled_df, best_params)

    
    plot_anomalies_2d(X_scaled_df, anomaly_predictions, feature_names)
    
    
    generate_shap_plots(final_model, X_scaled_df, feature_names)
    
    print("\n--- Enhanced Isolation Forest Analysis Complete ---")


if __name__ == "__main__":
    main()
