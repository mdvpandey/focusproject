"""
capture_and_feature_extract.py
Captures webcam frames, uses MediaPipe to extract face/eye landmarks,
computes simple per-frame features and logs them to CSV.

Usage:
    python src/capture_and_feature_extract.py --output /tmp/per_frame_features.csv --duration 30
Press 'q' to stop early.
"""
import cv2, argparse, csv, time, os
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh

def compute_eye_ratio(landmarks, left_idxs, right_idxs):
    lx = np.mean([landmarks[i].x for i in left_idxs])
    rx = np.mean([landmarks[i].x for i in right_idxs])
    cx = (lx+rx)/2.0
    return cx - 0.5

def eye_open_est(landmarks, upper_idx, lower_idx):
    return max(0.0, landmarks[upper_idx].y - landmarks[lower_idx].y)

def main(output, duration):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Exiting.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    mp_face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    fieldnames = ['timestamp','frame_idx','face_present','eye_center_offset','left_eye_open_est','right_eye_open_est']
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        frame_idx = 0
        start = time.time()
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            row = {'timestamp':time.time(),'frame_idx':frame_idx,'face_present':0,'eye_center_offset':0.0,'left_eye_open_est':0.0,'right_eye_open_est':0.0}
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                row['face_present'] = 1
                left_idxs = [33,133,160,159,158,157,173]
                right_idxs = [263,362,387,386,385,384,398]
                eye_offset = compute_eye_ratio(lm, left_idxs, right_idxs)
                row['eye_center_offset'] = float(eye_offset)
                row['left_eye_open_est'] = float(eye_open_est(lm,159,145))
                row['right_eye_open_est'] = float(eye_open_est(lm,386,374))
                # draw landmarks for simple feedback
                for pt in lm:
                    x = int(pt.x * w); y = int(pt.y * h)
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)
            writer.writerow(row)
            frame_idx += 1
            cv2.imshow('Capture (press q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Saved per-frame features to", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/mnt/data/per_frame_features.csv')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    args = parser.parse_args()
    main(args.output, args.duration)
