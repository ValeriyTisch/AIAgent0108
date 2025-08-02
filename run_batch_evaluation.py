import os
import pandas as pd
from src.pdf_llm_agent_pipeline import run_agent_on_text
from src.text_extraction import extract_text_from_pdf
from src.eval_utils import evaluate_output, export_results_to_excel

INPUT_XLSX = "data/test_ground_truth.xlsx"
PDF_DIR = "pdfs"

def main():
    df = pd.read_excel(INPUT_XLSX)
    all_results = []

    sum_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "accuracy": 0
    }
    count = 0

    for idx, row in df.iterrows():
        filename = row.get("filename")
        if not filename:
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        if not os.path.exists(pdf_path):
            print(f"⚠️ PDF не найден: {pdf_path}")
            continue

        try:
            text = extract_text_from_pdf(pdf_path)
            ground_truth = row.drop("filename").dropna().to_dict()

            prediction = run_agent_on_text(text, ground_truth)
            eval_metrics = evaluate_output(prediction, ground_truth)

            all_results.append({
                "filename": filename,
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                **{f"{k}_match": v["match"] for k, v in eval_metrics["details"].items()},
                **prediction
            })

            sum_metrics["precision"] += eval_metrics["precision"]
            sum_metrics["recall"] += eval_metrics["recall"]
            sum_metrics["f1"] += eval_metrics["f1"]
            sum_metrics["accuracy"] += eval_metrics["accuracy"]
            count += 1

        except Exception as e:
            print(f"❌ Ошибка при обработке {filename}: {e}")

    if count > 0:
        print("=== Summary metrics for all documents ===")
        print(f"Precision: {sum_metrics['precision'] / count:.3f}")
        print(f"Recall:    {sum_metrics['recall'] / count:.3f}")
        print(f"F1-score:  {sum_metrics['f1'] / count:.3f}")
        print(f"Accuracy:  {sum_metrics['accuracy'] / count:.3f}")

    export_results_to_excel(all_results)

if __name__ == "__main__":
    main()
