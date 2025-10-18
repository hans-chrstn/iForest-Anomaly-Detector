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
from sklearn.metrics import f1_score, make_scorer
from sklearn.decomposition import PCA
import shap
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
        return None, None, None

    if 'Target' in df.columns:
        features_df = df.drop('Target', axis=1)
    else:
        features_df = df

    feature_names = features_df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    print("Data loaded and scaled successfully.")
    return df, X_scaled_df, feature_names

def plot_sop1_consistency(X_scaled_df):
    print("\n--- SOP 1: Demonstrating Consistency Enhancement ---")

    np.random.seed(RANDOM_STATE)
    selected_samples_indices = np.random.choice(len(X_scaled_df), 5, replace=False)
    
    scores_variability = []
    for i in range(100):
        if_model_var = IsolationForest(random_state=i, contamination='auto')
        if_model_var.fit(X_scaled_df)
        scores = if_model_var.decision_function(X_scaled_df.iloc[selected_samples_indices])
        scores_variability.append(scores)
    
    scores_df_variability = pd.DataFrame(np.array(scores_variability), columns=[f'Sample {idx}' for idx in selected_samples_indices])
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=scores_df_variability)
    plt.title('Figure 1.2: Variability of Anomaly Scores (The Problem)')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.savefig(os.path.join(OUTPUT_FOLDER, "SOP1_Anomaly_Score_Variability_Enhanced.png"))
    print("Generated plot for SOP 1 (Problem): SOP1_Anomaly_Score_Variability_Enhanced.png")
    plt.close()

    if_model_consistent = IsolationForest(random_state=RANDOM_STATE, contamination='auto')
    if_model_consistent.fit(X_scaled_df)
    consistent_scores = []
    for _ in range(100):
        scores = if_model_consistent.decision_function(X_scaled_df.iloc[selected_samples_indices])
        consistent_scores.append(scores)
        
    scores_df_consistent = pd.DataFrame(np.array(consistent_scores), columns=[f'Sample {idx}' for idx in selected_samples_indices])
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=scores_df_consistent)
    plt.title('Figure 2.1: Consistency of Anomaly Scores with Fixed Random State (The Solution)')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.savefig(os.path.join(OUTPUT_FOLDER, "SOP1_Consistency_Solution_Enhanced.png"))
    print("Generated plot for SOP 1 (Solution): SOP1_Consistency_Solution_Enhanced.png")
    plt.close()

def optimize_and_plot_sop2(X_scaled_df, df_original):
    print("\n--- SOP 2: Finding Optimal Model Hyperparameters ---")
    
    y_eval = df_original['Fault_Type'].apply(lambda x: -1 if x > 0 else 1).values
    
    param_dist = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_features': [0.7, 0.8, 0.9, 1.0],
        'contamination': [0.01, 0.05, 0.1, 0.15],
        'max_samples': [0.7, 0.8, 0.9, 1.0, 'auto']
    }
    
    scorer = make_scorer(f1_score, pos_label=-1)
    
    iso_forest = IsolationForest(random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=iso_forest,
        param_distributions=param_dist,
        n_iter=50,
        scoring=scorer,
        cv=5,
        verbose=1,
        n_jobs=1,
        random_state=RANDOM_STATE
    )
    
    print("Running RandomizedSearchCV... (this may take a moment)")
    random_search.fit(X_scaled_df, y_eval)
    
    best_params = random_search.best_params_
    print(f"Hyperparameter optimization complete. Best Parameters Found: {best_params}")
    
    cv_results = pd.DataFrame(random_search.cv_results_)
    grouped_by_n = cv_results.groupby('param_n_estimators')['mean_test_score'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='param_n_estimators', y='mean_test_score', data=grouped_by_n, marker='o')
    plt.title(f'Figure 1.3: Performance (F1-score) vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean F1-score')
    plt.savefig(os.path.join(OUTPUT_FOLDER, "SOP2_F1_Score_vs_Estimators_Enhanced.png"))
    print("Generated plot for SOP 2: SOP2_F1_Score_vs_Estimators_Enhanced.png")
    plt.close()

    return best_params

def train_and_visualize_sop3(X_scaled_df, best_params, feature_names):
    print("\n--- SOP 3: Final Model Training, Visualization, and Interpretation ---")

    final_params = best_params.copy()
    final_params['random_state'] = RANDOM_STATE
    final_model = IsolationForest(**final_params)
    final_model.fit(X_scaled_df)
    anomaly_predictions = final_model.predict(X_scaled_df)
    
    print("Generating PCA plot...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled_df)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_predictions, cmap='coolwarm', s=100, alpha=0.8)
    plt.title('Figure 5.1: Final Anomaly Detection Result (PCA Plot)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ['Normal', 'Anomaly'], title="Status")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "SOP_Final_Anomaly_Detection_PCA_Enhanced.png"))
    print("Generated plot for SOP 3 (PCA): SOP_Final_Anomaly_Detection_PCA_Enhanced.png")
    plt.close()

    print("Generating SHAP plot...")
    explainer = shap.TreeExplainer(final_model)
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_scaled_df)
    
    plt.figure()
    shap.summary_plot(shap_values, X_scaled_df, feature_names=feature_names, show=False)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title('Figure 5.2: SHAP Feature Importance Summary', size=16)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "SHAP_Summary_Plot_Enhanced.png"))
    print("Generated plot for SOP 3 (SHAP): SHAP_Summary_Plot_Enhanced.png")
    plt.close()

def main():
    setup_environment()
    
    df_original, X_scaled_df, feature_names = load_and_preprocess_data(DATA_FILE_PATH)
    if df_original is None:
        return

    plot_sop1_consistency(X_scaled_df)
    
    best_params = optimize_and_plot_sop2(X_scaled_df, df_original)
    
    train_and_visualize_sop3(X_scaled_df, best_params, feature_names)
    
    print("\n--- Enhanced Isolation Forest Analysis (Full Equivalent) Complete ---")

if __name__ == "__main__":
    main()
