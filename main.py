import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Tracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = {}
    
    def update(self, detections):
        matched, unmatched_det, unmatched_trk = self._match(detections)
        
        for d, t in matched:
            self.tracks[t]['bbox'] = detections[d]
            self.tracks[t]['hits'] += 1
            self.tracks[t]['age'] = 0
        
        for d in unmatched_det:
            self.tracks[self.next_id] = {'bbox': detections[d], 'hits': 1, 'age': 0}
            self.next_id += 1
        
        for t in unmatched_trk:
            self.tracks[t]['age'] += 1
        
        self.tracks = {k: v for k, v in self.tracks.items() if v['age'] <= self.max_age}
        
        result = []
        for tid, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                result.append(np.append(track['bbox'][:4], tid))
        
        return np.array(result) if result else np.empty((0, 5))
    
    def _match(self, detections):
        if len(detections) == 0 or len(self.tracks) == 0:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        cost = cdist([t['bbox'][:4] for t in self.tracks.values()], 
                     detections[:, :4], metric='euclidean')
        cost[cost > 100] = 1e6
        
        row, col = linear_sum_assignment(cost)
        matched = [(col[i], list(self.tracks.keys())[row[i]]) for i in range(len(row))]
        unmatched_det = [i for i in range(len(detections)) if i not in col]
        unmatched_trk = [list(self.tracks.keys())[i] for i in range(len(self.tracks)) if i not in row]
        
        return matched, unmatched_det, unmatched_trk

class LicensePlateReader:
    def __init__(self, model_name='yolov8m.pt'):
        self.yolo = YOLO(model_name)
        self.ocr = easyocr.Reader(['en'], gpu=False)
        self.tracker = Tracker(max_age=30, min_hits=2)
        self.plate_history = defaultdict(list)
    
    def detect_plates(self, frame, conf=0.6):
        results = self.yolo(frame, conf=conf, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    if conf >= 0.6:
                        detections.append([*box, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def recognize_plate_multi(self, roi):
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return "", 0.0
        
        attempts = []
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        _, thresh = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        upscaled = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        try:
            results = self.ocr.readtext(upscaled, detail=1)
            if results:
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                plate_text = ''.join(texts).upper().strip()
                plate_text = ''.join(c for c in plate_text if c.isalnum())
                attempts.append((plate_text, np.mean(confs)))
        except:
            pass
        
        inverted = cv2.bitwise_not(upscaled)
        try:
            results = self.ocr.readtext(inverted, detail=1)
            if results:
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                plate_text = ''.join(texts).upper().strip()
                plate_text = ''.join(c for c in plate_text if c.isalnum())
                attempts.append((plate_text, np.mean(confs)))
        except:
            pass
        
        if attempts:
            best = max(attempts, key=lambda x: x[1])
            return best
        return "", 0.0
    
    def process_video(self, input_video, output_video):
        cap = cv2.VideoCapture(input_video)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            output_frame = frame.copy()
            
            detections = self.detect_plates(frame, conf=0.6)
            tracks = self.tracker.update(detections)
            
            for track in tracks:
                x1, y1, x2, y2, tid = map(int, track)
                
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.15)
                
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(w, x2 + pad_x)
                y2_pad = min(h, y2 + pad_y)
                
                box_width = x2_pad - x1_pad
                box_height = y2_pad - y1_pad
                if box_width < 40 or box_height < 20:
                    continue
                
                roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if roi.size > 0:
                    text, conf = self.recognize_plate_multi(roi)
                    
                    if text and conf > 0.45 and len(text) >= 4:
                        self.plate_history[tid].append((text, conf))
                        if len(self.plate_history[tid]) > 20:
                            self.plate_history[tid].pop(0)
                
                if self.plate_history[tid]:
                    weighted_votes = {}
                    for text, conf in self.plate_history[tid]:
                        if text not in weighted_votes:
                            weighted_votes[text] = 0
                        weighted_votes[text] += conf
                    best_text = max(weighted_votes.items(), key=lambda x: x[1])[0]
                else:
                    best_text = ""
                
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                id_text = f"ID: {tid}"
                id_font_size = 1.2
                id_thickness = 3
                id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, id_font_size, id_thickness)[0]
                cv2.rectangle(output_frame, (x1, y1 - 40), (x1 + id_text_size[0] + 10, y1 - 5), (0, 255, 0), -1)
                cv2.putText(output_frame, id_text, (x1 + 5, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, id_font_size, (0, 0, 0), id_thickness)
                
                if best_text:
                    plate_font_size = 2.0
                    plate_thickness = 4
                    plate_text_size = cv2.getTextSize(best_text, cv2.FONT_HERSHEY_SIMPLEX, plate_font_size, plate_thickness)[0]
                    cv2.rectangle(output_frame, (x1, y1 + 10), (x1 + plate_text_size[0] + 10, y1 + plate_text_size[1] + 20), (0, 0, 0), -1)
                    cv2.putText(output_frame, best_text, (x1 + 5, y1 + plate_text_size[1] + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, plate_font_size, (255, 255, 255), plate_thickness)
            
            cv2.putText(output_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(output_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        print(f"Output saved to {output_video}")

if __name__ == "__main__":
    reader = LicensePlateReader(model_name='yolov8m.pt')
    reader.process_video('traffic.mp4', 'output.mp4')