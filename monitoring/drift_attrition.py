import sys
import pandas as pd
from drift_utils import run_drift_report

def main():
    catalog = sys.argv[1] if len(sys.argv) > 1 else "ml_dev"
    reference_path = f"/Volumes/{catalog}/attrition_promotion/uploads/reference_train.csv"
    current_path = sys.argv[2]  # path to the most recent batch input CSV

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    feature_cols = [c for c in reference_df.columns if c != "target"]
    report = run_drift_report(reference_df, current_df, feature_cols)

    print(report.to_string(index=False))

    (spark.createDataFrame(report)
        .write.mode("append")
        .saveAsTable(f"{catalog}.attrition.drift_log"))

    if (report["status"] == "DRIFT_ALERT").any():
        print("DRIFT_ALERT triggered on one or more features")

if __name__ == "__main__":
    main()