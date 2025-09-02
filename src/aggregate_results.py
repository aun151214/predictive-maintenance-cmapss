# src/aggregate_results.py
import os
import glob
import pandas as pd
from fpdf import FPDF

RESULTS_DIR = "results"
OUTPUT_PDF = os.path.join(RESULTS_DIR, "final_report.pdf")
OUTPUT_MD = os.path.join(RESULTS_DIR, "final_report.md")

def load_results():
    """Load all CSV results and keep only valid evaluation rows (with RMSE/MAE/R2)."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    all_data = []
    for f in files:
        df = pd.read_csv(f)
        df["SourceFile"] = os.path.basename(f)
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError("No CSV result files found in results/")

    df = pd.concat(all_data, ignore_index=True)

    # ✅ Keep only rows with valid metrics (drop history NaN rows)
    df = df.dropna(subset=["rmse", "mae", "r2"], how="any")

    # ✅ Normalize column names (capitalized for report)
    df = df.rename(columns={
        "model": "Model",
        "dataset": "Dataset",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R2"
    })

    # ✅ Keep only the relevant columns
    df = df[["Model", "Dataset", "RMSE", "MAE", "R2"]]

    print(f"✅ Loaded {len(df)} evaluation results after cleaning")
    return df

def save_markdown(df):
    """Save results as a Markdown summary table."""
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# 📊 Model Performance Summary\n\n")
        f.write(df.to_markdown(index=False))
    print(f"📝 Markdown report generated: {OUTPUT_MD}")

def save_pdf(df):
    """Save results into a styled PDF report."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 20, "Predictive Maintenance Report", ln=True, align="C")

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Evaluation Metrics", ln=True, align="C")

    # Table header
    pdf.set_font("Arial", "B", 10)
    col_widths = [30, 30, 30, 30, 30]
    headers = ["Model", "Dataset", "RMSE", "MAE", "R2"]

    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, border=1, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", "", 9)
    for _, row in df.iterrows():
        pdf.cell(col_widths[0], 8, str(row["Model"]), border=1, align="C")
        pdf.cell(col_widths[1], 8, str(row["Dataset"]), border=1, align="C")
        pdf.cell(col_widths[2], 8, f"{row['RMSE']:.2f}", border=1, align="C")
        pdf.cell(col_widths[3], 8, f"{row['MAE']:.2f}", border=1, align="C")
        pdf.cell(col_widths[4], 8, f"{row['R2']:.3f}", border=1, align="C")
        pdf.ln()

    pdf.output(OUTPUT_PDF)
    print(f"📄 PDF report generated: {OUTPUT_PDF}")

def generate_report():
    df = load_results()
    save_markdown(df)
    save_pdf(df)

if __name__ == "__main__":
    generate_report()
