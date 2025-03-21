import pandas as pd
import matplotlib.pyplot as plt
import os

# ✅ Use the correct path for the Excel file
file_path = "/Users/anumulasanjana/Desktop/50_DATA.xlsx"

# Load the Excel file
df = pd.read_excel(file_path)

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Get the list of unique stocks
stocks = df["Stock"].unique()

# Create a folder for saving graphs
output_folder = "stock_graphs_50_DATA"
os.makedirs(output_folder, exist_ok=True)

# Generate and save plots for each stock
for stock in stocks:
    stock_data = df[df["Stock"] == stock]

    plt.figure(figsize=(8, 4))
    plt.plot(stock_data["Date"], stock_data["Actual Price"], label="Actual Price", marker="o", color="blue")
    plt.plot(stock_data["Date"], stock_data["Predicted Price"], label="Predicted Price", linestyle="dashed", marker="x", color="red")

    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.title(f"Actual vs. Predicted Prices for {stock}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # Save each graph as an image
    plot_path = os.path.join(output_folder, f"{stock}_prediction.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

print(f"\n✅ Graphs saved in the '{output_folder}' folder.")
