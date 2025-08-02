import pandas as pd
from typing import Dict

def evaluate_output(prediction: Dict, ground_truth: Dict) -> Dict:
    if not ground_truth:
        return {}

    metrics = {
        "total": 0,
        "matched": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "details": {}
    }

    for key, expected in ground_truth.items():
        actual = prediction.get(key)
        match = False

        if isinstance(expected, bool):
            match = (actual is True) if expected else (actual is False)
            if match and expected:
                metrics["TP"] += 1
            elif not match and actual:
                metrics["FP"] += 1
            elif not match and expected:
                metrics["FN"] += 1
        elif expected is not None:
            match = str(actual).strip() == str(expected).strip()
            if match:
                metrics["TP"] += 1
            else:
                # Для не булевых метрик считаем FP и FN одинаково
                metrics["FP"] += 1
                metrics["FN"] += 1

        metrics["total"] += 1
        metrics["matched"] += int(match)
        metrics["details"][key] = {
            "expected": expected,
            "actual": actual,
            "match": match
        }

    precision = metrics["TP"] / (metrics["TP"] + metrics["FP"]) if (metrics["TP"] + metrics["FP"]) > 0 else 0
    recall = metrics["TP"] / (metrics["TP"] + metrics["FN"]) if (metrics["TP"] + metrics["FN"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": metrics["matched"] / metrics["total"] if metrics["total"] else 0
    })

    return metrics

def export_results_to_excel(results: list, path: str = "results/predictions_with_metrics.xlsx"):
    """
    Сохраняет список словарей (по одному на документ) в Excel-файл
    """
    df = pd.DataFrame(results)
    df.to_excel(path, index=False)
    print(f"📤 Результаты сохранены в {path}")
