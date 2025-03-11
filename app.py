import os
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import uuid
from sklearn.cluster import KMeans # type: ignore
from flask import Flask, render_template, request # type: ignore

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SEGMENTATION_TYPES = {
    "Demographic": ["Annual Income (k$)", "Age"],
    "Behavioral": ["Spending Score (1-100)", "Annual Income (k$)"],
    "Psychographic": ["Spending Score (1-100)", "Age"],
    "Geographic": ["Latitude", "Longitude"]
}

def clear_old_plots():
    """Remove old scatter plots to prevent caching issues."""
    for file in os.listdir(UPLOAD_FOLDER):
        if file.startswith("scatter_"):
            os.remove(os.path.join(UPLOAD_FOLDER, file))

def perform_clustering(file_path, segmentation_type):
    """Load CSV, apply K-Means clustering based on selected segmentation type."""
    df = pd.read_csv(file_path)

    if segmentation_type not in SEGMENTATION_TYPES:
        raise ValueError("Invalid segmentation type selected.")

    required_columns = SEGMENTATION_TYPES[segmentation_type]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column in dataset: {col}")

    X = df[required_columns]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    return df

def generate_scatter_plot(df, segmentation_type):
    """Generate scatter plot for selected segmentation type."""
    plt.figure(figsize=(8, 5))

    x_col, y_col = SEGMENTATION_TYPES[segmentation_type]
    df["Cluster"] = df["Cluster"].astype(int)

    clusters = df["Cluster"].unique()
    for cluster in clusters:
        cluster_data = df[df["Cluster"] == cluster]
        plt.scatter(cluster_data[x_col], cluster_data[y_col], label=f"Cluster {cluster}")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Customer Segmentation - {segmentation_type}")
    plt.legend()

    plot_filename = f"scatter_{segmentation_type}_{uuid.uuid4().hex[:8]}.png"
    scatter_plot_path = os.path.join(app.config["UPLOAD_FOLDER"], plot_filename)
    plt.savefig(scatter_plot_path)
    plt.close()

    return plot_filename

@app.route("/", methods=["GET", "POST"])
def index():
    scatter_plot = None
    selected_type = None

    if request.method == "POST":
        file = request.files["file"]
        selected_type = request.form["segmentation_type"]

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.csv")
            file.save(file_path)

            clear_old_plots()  # Remove old plots
            df = perform_clustering(file_path, selected_type)
            scatter_plot = generate_scatter_plot(df, selected_type)

    return render_template("index.html", segmentation_types=SEGMENTATION_TYPES.keys(), 
                           scatter_plot=scatter_plot, selected_type=selected_type)

if __name__ == "__main__":
    app.run(debug=True)
