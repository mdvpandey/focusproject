"""
predict_live.py
Runs live capture for `duration` seconds, aggregates into windows and predicts focused/distracted using trained model.
Usage:
    python src/predict_live.py --model /path/to/focus_model.joblib --duration 30 --window 10
"""
import argparse, time, joblib, os, tempfile, sys, subprocess
from src import capture_and_feature_extract as capmod  # when run as package, ensure PYTHONPATH or run from BASE

# For portability, we'll call the capture script as a subprocess then aggregate and predict
def run_capture(output, duration):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'capture_and_feature_extract.py'), '--output', output, '--duration', str(duration)]
    print("Running capture:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_aggregate(input_csv, output_csv, window):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'window_aggregator.py'), '--input', input_csv, '--output', output_csv, '--window', str(window)]
    print("Aggregating windows:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def predict_windows(model_path, windows_csv):
    import pandas as pd
    clf = joblib.load(model_path)
    df = pd.read_csv(windows_csv)
    feature_cols = [c for c in df.columns if c not in ('window_id','label')]
    preds = clf.predict(df[feature_cols].fillna(0))
    df['predicted_focus'] = preds
    score = (df['predicted_focus'].sum() / max(1,len(df))) * 100.0
    print("Window predictions:")
    print(df[['window_id','predicted_focus']])
    print(f"Focus score: {score:.1f}%")
    return df, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()
    tmp_perframe = os.path.join('/tmp','per_frame_features.csv')
    tmp_windows = os.path.join('/tmp','window_features.csv')
    run_capture(tmp_perframe, args.duration)
    run_aggregate(tmp_perframe, tmp_windows, args.window)
    predict_windows(args.model, tmp_windows)
