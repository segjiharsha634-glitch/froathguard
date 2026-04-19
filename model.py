import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# STEP 1: GENERATE SIMULATED DATA
# ==========================================
def generate_flotation_data(n_samples=1440):
    """Simulates 24 hours (1440 mins) of continuous flotation circuit data."""
    print("1. Generating simulated plant data...")
    time = np.arange(n_samples)
    
    df = pd.DataFrame({
        "Air_Flow_Rate_m3h": 150 + np.sin(time/50)*10 + np.random.normal(0, 2, n_samples),
        "Reagent_Dosage_gh": 45 + np.random.normal(0, 1.5, n_samples),
        "Pulp_Density_pct": 32 + np.cos(time/100)*2 + np.random.normal(0, 0.5, n_samples),
        "Froth_Depth_cm": 25 + np.sin(time/75)*3 + np.random.normal(0, 1, n_samples),
        "pH_Level": 8.5 + np.random.normal(0, 0.1, n_samples),
        "Agitator_Power_kW": 200 + np.random.normal(0, 5, n_samples)
    })
    
    # Inject Fault: Reagent Blockage at t=800 to 950
    fault_start, fault_end = 800, 950
    df.loc[fault_start:fault_end, "Reagent_Dosage_gh"] -= 15 
    decay = np.linspace(0, 10, fault_end - fault_start + 1)
    df.loc[fault_start:fault_end, "Froth_Depth_cm"] -= decay
    df.loc[fault_start:fault_end, "Air_Flow_Rate_m3h"] += np.random.normal(0, 15, fault_end - fault_start + 1)
    

    df.to_csv("simulated_flotation_data.csv", index=False)
    return df

# ==========================================
# STEP 2: MACHINE LEARNING PIPELINE (NO TENSORFLOW)
# ==========================================
def run_ml_pipeline(df, sensor_cols):
    print("2. Training Machine Learning models...")
    
    # 2a. Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[sensor_cols])
    
    # 2b. Train Isolation Forest
    print("   -> Running Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_scaled)
    # Convert scores so higher = more anomalous
    raw_if_scores = -iso_forest.score_samples(X_scaled)
    y_scores_if = (raw_if_scores - raw_if_scores.min()) / (raw_if_scores.max() - raw_if_scores.min())
    threshold_if = 0.55 
    
    # 2c. Train PCA (Replaces the Autoencoder)
    print("   -> Running PCA Anomaly Detector...")
    # Compress the 6 sensors down to 3 main components (the "bottleneck")
    pca = PCA(n_components=3, random_state=42)
    
    # Train only on the first 600 minutes (Normal operations)
    X_train = X_scaled[:600] 
    pca.fit(X_train)
    
    # Reconstruct the entire dataset
    X_projected = pca.transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_projected)
    
    # Calculate Reconstruction Error (Mean Squared Error)
    mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
    
    # Normalize PCA scores to roughly [0, 1] for the dashboard
    y_scores_pca = mse / np.max(mse)
    threshold_pca = 0.20 

    return y_scores_if, y_scores_pca, threshold_if, threshold_pca

# ==========================================
# STEP 3: DASHBOARD GENERATION
# ==========================================
def create_dashboard(df, sensor_cols, y_scores_if, y_scores_pca, threshold_if, threshold_pca):
    """Build and save a Plotly multi-panel dashboard."""
    print("3. Generating interactive dashboard...")
    x = np.arange(len(df))

    specs = []
    for r in range(9):
        if r < 6:
            if r == 0:
                specs.append([{"rowspan": 1}, {"rowspan": 6}])
            else:
                specs.append([{"rowspan": 1}, None])
        else:
            specs.append([{}, {}])

    fig = make_subplots(
        rows=9, cols=2, shared_xaxes=False,
        specs=specs, subplot_titles=(list(sensor_cols) + ["Anomaly Scores"])
    )

    for i, sensor in enumerate(sensor_cols):
        row = i + 1
        fig.add_trace(go.Scatter(x=x, y=df[sensor], mode="lines", name=sensor), row=row, col=1)

    fig.add_trace(go.Scatter(x=x, y=y_scores_if, name="Isolation Forest score"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=y_scores_pca, name="PCA score", line=dict(color="#ff7f0e")), row=1, col=2)

    fig.add_hline(y=threshold_if, line_dash="dash", row=1, col=2, annotation_text=f"IF thr={threshold_if:.2f}")
    fig.add_hline(y=threshold_pca, line_dash="dot", row=1, col=2, annotation_text=f"PCA thr={threshold_pca:.2f}", line_color="#ff7f0e")

    # Add red shaded box indicating the actual fault period
    fig.add_vrect(x0=800, x1=950, fillcolor="red", opacity=0.1, layer="below", line_width=0, row=1, col=2, annotation_text="Actual Fault", annotation_position="top left")

    fig.update_layout(height=1200, title_text="Froth Flotation Anomaly Detection Dashboard")
    return fig

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    sensors = [
        "Air_Flow_Rate_m3h", "Reagent_Dosage_gh", 
        "Pulp_Density_pct", "Froth_Depth_cm", 
        "pH_Level", "Agitator_Power_kW"
    ]
    
    plant_data = generate_flotation_data()
    
    if_scores, pca_scores, if_thr, pca_thr = run_ml_pipeline(plant_data, sensors)
    
    dashboard_fig = create_dashboard(plant_data, sensors, if_scores, pca_scores, if_thr, pca_thr)
    
    output_file = "anomaly_dashboard.html"
    dashboard_fig.write_html(output_file)
    print(f"4. Success! Dashboard saved as '{output_file}'.")