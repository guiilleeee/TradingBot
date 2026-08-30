"""
analyze_gemini.py — One-time analysis script for the 7-day Gemini comparison test.
Compares gemini_test_signals.csv to the production signals.csv.
"""
import pandas as pd
from pathlib import Path
import sys

def main():
    prod_csv = Path("signals.csv")
    gemini_csv = Path("gemini_test_signals.csv")

    if not prod_csv.exists() or not gemini_csv.exists():
        print("Both signals.csv and gemini_test_signals.csv must exist to run analysis.")
        sys.exit(1)

    try:
        df_prod = pd.read_csv(prod_csv)
        df_gemini = pd.read_csv(gemini_csv)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit(1)
        
    if df_prod.empty or df_gemini.empty:
        print("One or both CSV files are empty.")
        sys.exit(0)

    # Convert timestamps to datetime to allow for approximate matching
    # In practice, they run on the same schedule, but maybe off by a few minutes
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"])
    df_gemini["timestamp"] = pd.to_datetime(df_gemini["timestamp"])
    
    # We need to extract the final action and confidence. The CSV has 'final_signal' as JSON string
    import json
    
    def extract_metrics(df):
        actions = []
        confidences = []
        for raw_json in df["final_signal"]:
            try:
                data = json.loads(raw_json)
                actions.append(data.get("action", "hold"))
                confidences.append(data.get("confidence", 0.0))
            except Exception:
                actions.append("hold")
                confidences.append(0.0)
        df["action"] = actions
        df["confidence"] = confidences
        return df

    df_prod = extract_metrics(df_prod)
    df_gemini = extract_metrics(df_gemini)

    # Sort and merge_asof based on timestamp and symbol
    df_prod = df_prod.sort_values("timestamp")
    df_gemini = df_gemini.sort_values("timestamp")
    
    # We can do a near join
    merged = pd.merge_asof(
        df_gemini, 
        df_prod, 
        on="timestamp", 
        by="symbol", 
        direction="nearest", 
        tolerance=pd.Timedelta("1 hour"),
        suffixes=("_gemini", "_prod")
    )
    
    # Filter rows where both models generated a signal
    merged = merged.dropna(subset=["action_prod", "action_gemini"])
    
    if merged.empty:
        print("No overlapping signals found between the two models.")
        sys.exit(0)

    print("=== Gemini 7-Day Comparison Results ===")
    
    symbols = merged["symbol"].unique()
    for symbol in symbols:
        print(f"\n--- Symbol: {symbol} ---")
        df_sym = merged[merged["symbol"] == symbol]
        
        # Action counts
        prod_counts = df_sym["action_prod"].value_counts().to_dict()
        gemini_counts = df_sym["action_gemini"].value_counts().to_dict()
        
        print("Action Counts (Claude / Gemini):")
        for action in ["buy", "sell", "hold"]:
            p_cnt = prod_counts.get(action, 0)
            g_cnt = gemini_counts.get(action, 0)
            print(f"  {action.upper()}: {p_cnt} / {g_cnt}")
            
        # Average Confidence
        avg_conf_prod = df_sym["confidence_prod"].mean()
        avg_conf_gemini = df_sym["confidence_gemini"].mean()
        print(f"Average Confidence: Claude = {avg_conf_prod:.2f} | Gemini = {avg_conf_gemini:.2f}")
        
        # Disagreements
        disagreements = df_sym[df_sym["action_prod"] != df_sym["action_gemini"]]
        print(f"Disagreements: {len(disagreements)} / {len(df_sym)} cycles")
        
        if len(disagreements) > 0:
            print("  Sample disagreements:")
            for _, row in disagreements.head(3).iterrows():
                print(f"    {row['timestamp']} -> Claude: {row['action_prod']}, Gemini: {row['action_gemini']}")

if __name__ == "__main__":
    main()
