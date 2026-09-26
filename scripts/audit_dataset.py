import os
import cv2
from engine.detector import SonarDetector

detector = SonarDetector('models/best_detector.onnx')

dirs = ['data/samples', 'sample_sonar_data/positives/test/images', 'sample_sonar_data/test/images']
seen = set()

print(f"{'IMAGE':<48} | {'SIZE':<10} | {'DETS':<5} | {'PREDICTIONS (CONF)'}")
print('-' * 95)

shipwreck_counts = 0
total_detections = 0
class_histogram = {}

for d in dirs:
    if not os.path.exists(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        p = os.path.join(d, f)
        canon = os.path.basename(f)
        if canon in seen:
            continue
        seen.add(canon)
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets, _ = detector.detect(img, conf_threshold=0.25, tiling=True)
        total_detections += len(dets)
        
        det_strs = []
        for x in dets:
            c = x['class']
            conf = x['confidence']
            det_strs.append(f"{c} ({conf*100:.1f}%)")
            class_histogram[c] = class_histogram.get(c, 0) + 1
            if c == 'shipwreck':
                shipwreck_counts += 1
                
        pred_summary = ", ".join(det_strs) if det_strs else "[NONE (Clean Background)]"
        print(f"{canon:<48} | {w}x{h:<6} | {len(dets):<5} | {pred_summary}")

print('-' * 95)
print(f"Total Unique Images Evaluated: {len(seen)}")
print(f"Total Detections at conf >= 0.25: {total_detections}")
print(f"Class Distribution: {class_histogram}")
print(f"Total Shipwreck Detections: {shipwreck_counts} ({shipwreck_counts/max(1, total_detections)*100:.1f}% of all detections)")
