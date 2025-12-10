"""
window_aggregator.py
Aggregates per-frame csv into time-window features (e.g., 10-second windows).
Produces a CSV with one row per window and optional label column for training.
Usage:
    python src/window_aggregator.py --input /tmp/per_frame_features.csv --output /tmp/window_features.csv --window 10
"""
import pandas as pd, argparse, math, os

def aggregate(input_csv, output_csv, window_seconds):
    df = pd.read_csv(input_csv)
    if df.empty:
        print("Input CSV is empty:", input_csv)
        return
    df['ts_rel'] = df['timestamp'] - df['timestamp'].iloc[0]
    df['window_id'] = (df['ts_rel'] // window_seconds).astype(int)
    agg = df.groupby('window_id').agg(
        face_present_ratio = ('face_present','mean'),
        mean_eye_offset = ('eye_center_offset','mean'),
        std_eye_offset = ('eye_center_offset','std'),
        mean_left_eye_open = ('left_eye_open_est','mean'),
        mean_right_eye_open = ('right_eye_open_est','mean'),
        frames = ('frame_idx','count')
    ).reset_index()
    agg['frames'] = agg['frames'].fillna(0).astype(int)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    agg.to_csv(output_csv, index=False)
    print("Saved", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/data/per_frame_features.csv')
    parser.add_argument('--output', default='/mnt/data/window_features.csv')
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()
    aggregate(args.input, args.output, args.window)
