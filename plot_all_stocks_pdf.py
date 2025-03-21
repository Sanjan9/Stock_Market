

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load the CSV file
file_path = "backtest_50_companies.csv"  # Ensure it's in the same folder as this script
backtest_results = pd.read_csv(file_path)

# Convert Date column to datetime format
backtest_results["Date"] = pd.to_datetime(backtest_results["Date"])

# Get the list of unique stocks
stocks = backtest_results["Stock"].unique()

# Create a single PDF file
pdf_filename = "stock_predictions.pdf"
with PdfPages(pdf_filename) as pdf:
    for stock in stocks:
        stock_data = backtest_results[backtest_results["Stock"] == stock]

        # Create the plot
        plt.figure(figsize=(8, 4))
        plt.plot(stock_data["Date"], stock_data["Actual Price"], label="Actual Price", marker="o", color="blue")
        plt.plot(stock_data["Date"], stock_data["Predicted Price"], label="Predicted Price", linestyle="dashed", marker="x", color="red")

        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.title(f"Actual vs. Predicted Prices for {stock}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # Save the plot to the PDF
        pdf.savefig()
        plt.close()

print(f"\n All graphs have been saved in '{pdf_filename}'.")
