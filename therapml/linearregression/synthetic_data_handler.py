import os
import torch
import pandas as pd
import matplotlib.pyplot as plt


def generate_synthetic_data(
    n_samples=100,
    weight=3.5,
    bias=2.0,
    noise_std=1.0,
    data_dir="data",
    filename="data.csv",
    plot_dir="plots"
):
    """
    Generates synthetic linear regression data:
        y = weight * x + bias + noise

    Saves:
        - CSV file inside data_dir
        - Scatter plot inside plot_dir
    """

    # Set reproducibility
    torch.manual_seed(42)

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Generate input feature
    x = torch.linspace(-10, 10, n_samples).unsqueeze(1)

    # Generate noise
    noise = torch.randn(n_samples, 1) * noise_std

    # Linear relationship
    y = weight * x + bias + noise

    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "x": x.squeeze().numpy(),
        "y": y.squeeze().numpy()
    })

    # Save CSV in data folder
    csv_path = os.path.join(data_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # Plot data
    plt.figure()
    plt.scatter(x.numpy(), y.numpy())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Linear Regression Data")

    plot_path = os.path.join(plot_dir, "synthetic_data_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

    return x, y


def load_data_from_csv(data_dir="data", filename="data.csv"):
    """
    Loads dataset from CSV file (from data_dir) and returns PyTorch tensors.
    """

    csv_path = os.path.join(data_dir, filename)

    df = pd.read_csv(csv_path)

    x = torch.tensor(df["x"].values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

    return x, y


if __name__ == "__main__":
    # Generate dataset
    generate_synthetic_data()

    # Load dataset
    x_loaded, y_loaded = load_data_from_csv()

    print("Loaded data shapes:")
    print("x:", x_loaded.shape)
    print("y:", y_loaded.shape)