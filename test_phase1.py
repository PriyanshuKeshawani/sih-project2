import os
import time
import cv2
from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.reflex import System1ReflexEngine

def run_tests():
    detector = SonarDetector('models/best_detector.onnx')
    physics = SonarPhysicsEngine()
    reflex = System1ReflexEngine()

    test_cases = [
        ('Background (Clear seabed)', 'data/samples/bg_1693569243.750_x2500.jpg'),
        ('Ghost Net', 'data/samples/synth_ghost_net_00001.png'),
        ('Shipwreck', 'data/samples/wreckR_ship-081_png.rf.6cf386b75ddb8ead86c0453021279296.jpg'),
        ('Submarine Pipeline', 'data/samples/pipe_1693569383.780_x3500.jpg'),
        ('Mine / Cylinder', 'data/samples/mine_0001_2015.jpg')
    ]

    print("=== MULTI-CLASS INFERENCE VERIFICATION ===")
    for label, path in test_cases:
        t0 = time.perf_counter()
        img = cv2.imread(path)
        if img is None:
            print(f"FAILED to read {path}")
            continue
        h, w = img.shape[:2]
        
        t_inf_start = time.perf_counter()
        detections, annotated = detector.detect(img, conf_threshold=0.25)
        t_inf_end = time.perf_counter()
        
        total_time_ms = (t_inf_end - t0) * 1000
        inf_time_ms = (t_inf_end - t_inf_start) * 1000
        
        print(f"\n--- {label} ({path}) ---")
        print(f"Dimensions: {w}x{h} px | Inference: {inf_time_ms:.1f}ms | Total: {total_time_ms:.1f}ms")
        print(f"Detections count: {len(detections)}")
        for d in detections:
            elev = physics.calculate_elevation(d['box'], w, h, altitude=12.0)
            geo = physics.georeference(d['box'], w, h)
            ref = reflex.evaluate_reflex(d, elev)
            print(f"  -> Class: {d['class']} | Conf: {d['confidence']*100:.1f}% | Box: {d['box']}")
            print(f"     Elevation: {elev}m | Geo: {geo['lat']}, {geo['lon']} | Reflex: {ref['decision_primitive']}")

if __name__ == '__main__':
    run_tests()
