"""
train_model.py
Trains a RandomForest classifier on window-level features CSV.
Usage:
    python src/train_model.py --input sample_data/window_features_labeled.csv --output /tmp/focus_model.joblib
"""
import pandas as pd, argparse, joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train(input_csv, model_out):
    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c not in ('window_id','label')]
    X = df[feature_cols].fillna(0)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print("Model saved to", model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/data/window_features_labeled.csv')
    parser.add_argument('--output', default='/mnt/data/focus_model.joblib')
    args = parser.parse_args()
    train(args.input, args.output)
