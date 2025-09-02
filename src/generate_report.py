import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

RESULTS_DIR = "results"
PDF_PATH = "summary_report.pdf"

def load_results():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*_results.csv"))
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            model = os.path.basename(file).split("_")[0]
            dataset = os.path.basename(file).split("_")[1]
            df["Model"] = model.upper()
            df["Dataset"] = dataset
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def plot_metrics(df, dataset):
    charts = []
    subset = df[df["Dataset"] == dataset]
    metrics = ["RMSE", "MAE", "R2"]

    for metric in metrics:
        plt.figure(figsize=(5,3))
        plt.bar(subset["Model"], subset[metric], color="skyblue")
        plt.title(f"{dataset} - {metric} by Model")
        plt.ylabel(metric)
        chart_path = os.path.join(RESULTS_DIR, f"{dataset}_{metric}.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        charts.append(chart_path)
    return charts

def plot_loss_curves(dataset):
    charts = []
    history_files = glob.glob(os.path.join(RESULTS_DIR, f"*_{dataset}_history.csv"))

    for file in history_files:
        try:
            df = pd.read_csv(file)
            base = os.path.basename(file).replace("_history.csv", "")
            model, dataset = base.split("_", 1)

            plt.figure(figsize=(5,3))
            plt.plot(df["epoch"], df["loss"], label="Train Loss")
            plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
            plt.title(f"Training Curve: {model.upper()} - {dataset}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            chart_path = os.path.join(RESULTS_DIR, f"{base}_loss.png")
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()
            charts.append(chart_path)
        except Exception as e:
            print(f"⚠️ Could not plot {file}: {e}")
    return charts

def generate_pdf(df):
    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("CMAPSS Predictive Maintenance Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    for dataset in df["Dataset"].unique():
        elements.append(Paragraph(f"Dataset: {dataset}", styles["Heading1"]))
        elements.append(Spacer(1, 12))

        # Table of results
        subset = df[df["Dataset"] == dataset][["Model", "RMSE", "MAE", "R2"]]
        data = [subset.columns.to_list()] + subset.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 15))

        # Metric bar charts
        elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
        charts = plot_metrics(df, dataset)
        for chart in charts:
            elements.append(Image(chart, width=400, height=250))
            elements.append(Spacer(1, 12))

        # Training loss curves
        elements.append(Paragraph("Training Loss Curves", styles["Heading2"]))
        loss_charts = plot_loss_curves(dataset)
        for chart in loss_charts:
            elements.append(Image(chart, width=400, height=250))
            elements.append(Spacer(1, 12))

        # Page break between datasets
        elements.append(PageBreak())

    doc.build(elements)
    print(f"✅ PDF report generated at {PDF_PATH}")

if __name__ == "__main__":
    df = load_results()
    if df.empty:
        print("⚠️ No results found in 'results/' folder.")
    else:
        generate_pdf(df)
