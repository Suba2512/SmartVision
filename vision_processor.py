import cv2
import numpy as np
from ultralytics import YOLO

class VisionProcessor:
    def __init__(self, model_name="yolov8n.pt"):
        # Load the lightweight YOLOv8n model
        # Note: If the model weights are not local, Ultralytics downloads them to the current directory or cache.
        self.model = YOLO(model_name)
        
        # Bilingual Dictionary (English YOLO label -> Tamil script & Tamil Phonetic)
        # Includes required minimum items and additional indoor targets for robustness.
        self.translation_dict = {
            # Objects
            "person": {"tamil": "மனிதர்", "phonetic": "Manithar"},
            "chair": {"tamil": "நாற்காலி", "phonetic": "Naarkali"},
            "bottle": {"tamil": "பாட்டில்", "phonetic": "Baattil"},
            "cell phone": {"tamil": "கைபேசி", "phonetic": "Kaibesi"},
            "backpack": {"tamil": "பை", "phonetic": "Bai"},
            "bed": {"tamil": "கட்டில்", "phonetic": "Kattil"},
            "dining table": {"tamil": "மேஜை", "phonetic": "Mejai"},
            "laptop": {"tamil": "மடிக்கணினி", "phonetic": "Madikkanini"},
            "door": {"tamil": "கதவு", "phonetic": "Kathavu"},
            "tv": {"tamil": "தொலைக்காட்சி", "phonetic": "Tholaikkaatchi"},
            "remote": {"tamil": "தொலைக்கட்டுப்பாடு", "phonetic": "Tholaikkattuppaadu"},
            "book": {"tamil": "புத்தகம்", "phonetic": "Puththagam"},
            "cup": {"tamil": "கோப்பை", "phonetic": "Koppai"},
            "couch": {"tamil": "சோபா", "phonetic": "Soba"},
            "keyboard": {"tamil": "விசைப்பலகை", "phonetic": "Visaippalagai"},
            "mouse": {"tamil": "சுட்டி", "phonetic": "Sutti"},
            "handbag": {"tamil": "கைப்பை", "phonetic": "Kaipai"},
            "suitcase": {"tamil": "பயணப்பெட்டி", "phonetic": "Payanappetti"},
            
            # Positions (Horizontal Zoning)
            "Left": {"tamil": "உங்களுக்கு இடது பக்கத்தில்", "phonetic": "ungalukku idathu pakkathil"},
            "Center": {"tamil": "உங்களுக்கு நேராக", "phonetic": "ungalukku neraaga"},
            "Right": {"tamil": "உங்களுக்கு வலது பக்கத்தில்", "phonetic": "ungalukku valathu pakkathil"},
            
            # Proximity/Distance
            "Close": {"tamil": "அருகில் உள்ளது", "phonetic": "arugil ullathu"},
            "Far": {"tamil": "தொலைவில் உள்ளது", "phonetic": "tholaivil ullathu"}
        }

    def process_frame(self, frame, conf_threshold=0.45):
        """
        Runs object detection on the input frame.
        Calculates zoning, calculates distance estimation ratios,
        draws premium HUD aesthetics and visual indicators.
        Returns the annotated frame and list of detected objects sorted by confidence.
        """
        h, w, _ = frame.shape
        
        # 1. Horizontal Zoning Divisions
        left_boundary = w // 3
        right_boundary = (2 * w) // 3
        
        # Draw translucent vertical zones guidelines
        self.draw_translucent_zone_lines(frame, left_boundary, right_boundary, h, w)

        # 2. Run YOLOv8 model inference
        # verbose=False prevents console logging spam from YOLO to keep terminal clean
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        
        # Iterate over detections
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            
            # Extract coordinates (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Calculate midpoint X-coordinate
            mid_x = (x1 + x2) // 2
            
            # Determine horizontal zone based on midpoint X
            if mid_x < left_boundary:
                zone = "Left"
            elif mid_x < right_boundary:
                zone = "Center"
            else:
                zone = "Right"
                
            # Proximity calculation: Box height / total frame height ratio
            box_height = y2 - y1
            height_ratio = box_height / h
            proximity = "Close" if height_ratio >= 0.45 else "Far"
            
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "zone": zone,
                "proximity": proximity
            })
            
        # Sort objects by confidence descending to get the highest confidence item first
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # 3. Draw bounding boxes and details on frame
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            zone = det["zone"]
            proximity = det["proximity"]
            
            # Use color-coding (Vibrant Green for CLOSE obstacles, Bright Cyan/Blue for FAR obstacles)
            color = (46, 204, 113) if proximity == "Close" else (52, 152, 219) # BGR: Green vs Sky Blue
            
            # Draw sleek bounding box with slightly thicker borders
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # Draw target corners for a modern HUD look
            self.draw_corner_brackets(frame, x1, y1, x2, y2, color)
            
            # Display tracking metrics (Name, Zone, Proximity) on screen in real time
            metric_text = f"{label.capitalize()} ({int(conf * 100)}%) - {zone} | {proximity}"
            
            # Text container background
            text_size = cv2.getTextSize(metric_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            bg_x2 = min(x1 + text_size[0] + 12, w)
            bg_y1 = max(y1 - 22, 0)
            bg_y2 = y1
            
            # Translucent label block
            self.draw_translucent_rect(frame, x1, bg_y1, bg_x2, bg_y2, color, alpha=0.7)
            cv2.putText(frame, metric_text, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
        return frame, detections

    def format_voice_cues(self, detection):
        """
        Formulates instructional English and Tamil speech phrases.
        - English format: "[Article] [Object] is [Close/Far] to/from you, [Position]."
        - Tamil format: "[Object] [Position] [Distance]."
        """
        label = detection["label"]
        zone = detection["zone"]
        proximity = detection["proximity"]
        
        # 1. English Instruction Construction
        # Determine proper article (a/an)
        first_char = label[0].lower() if label else ''
        article = "An" if first_char in ['a', 'e', 'i', 'o', 'u'] else "A"
        
        # Determine position description
        if zone == "Center":
            pos_desc = "straight ahead"
        elif zone == "Left":
            pos_desc = "on your left side"
        else:  # Right
            pos_desc = "on your right side"
            
        # Determine proximity description
        if proximity == "Close":
            prox_desc = "is close to you"
        else:
            prox_desc = "is far from you"
            
        english_phrase = f"{article} {label} {prox_desc}, {pos_desc}."
        
        # 2. Tamil Instruction Construction (Unicode & Phonetic)
        # Position Translation
        pos_data = self.translation_dict.get(zone, {"tamil": zone, "phonetic": zone})
        pos_ta = pos_data["tamil"]
        pos_ph = pos_data["phonetic"]
        
        # Distance Translation
        dist_data = self.translation_dict.get(proximity, {"tamil": proximity, "phonetic": proximity})
        dist_ta = dist_data["tamil"]
        dist_ph = dist_data["phonetic"]
        
        # Object Translation (with fallback to English label if missing)
        obj_data = self.translation_dict.get(label)
        if obj_data:
            obj_ta = obj_data["tamil"]
            obj_ph = obj_data["phonetic"]
        else:
            obj_ta = label
            obj_ph = label
            
        # Compile Tamil sentences: [Object] [Position] [Distance]
        # Example: "நாற்காலி உங்களுக்கு இடது பக்கத்தில் அருகில் உள்ளது."
        # Phonetic: "Naarkali ungalukku idathu pakkathil arugil ullathu."
        tamil_unicode = f"{obj_ta} {pos_ta} {dist_ta}."
        tamil_phonetic = f"{obj_ph} {pos_ph} {dist_ph}."
        
        return english_phrase, tamil_unicode, tamil_phonetic

    # UI / Design Utilities
    def draw_translucent_rect(self, img, x1, y1, x2, y2, color, alpha=0.4):
        """Draws a blended colored rectangle to represent overlay transparency."""
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size == 0:
            return
        rect = np.full(sub_img.shape, color, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0)
        img[y1:y2, x1:x2] = res

    def draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=15, thickness=3):
        """Draws small framing brackets at the bounding box corners to look high-tech."""
        # Top-Left Corner
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, lineType=cv2.LINE_AA)
        
        # Top-Right Corner
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, lineType=cv2.LINE_AA)
        
        # Bottom-Left Corner
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, lineType=cv2.LINE_AA)
        
        # Bottom-Right Corner
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, lineType=cv2.LINE_AA)

    def draw_translucent_zone_lines(self, frame, left_b, right_b, h, w):
        """Draws clean zone separator lines and headers at the bottom of the zones."""
        # Draw translucent separator columns
        cv2.line(frame, (left_b, 0), (left_b, h), (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (right_b, 0), (right_b, h), (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # Draw zone labels at the bottom edge with a dark semi-transparent panel
        panel_h = 30
        self.draw_translucent_rect(frame, 0, h - panel_h, w, h, (30, 30, 30), alpha=0.6)
        
        # Zone Text Label positioning
        zones = [
            ("LEFT / IDATHU", 10),
            ("CENTER / MAIYAM", left_b + 10),
            ("RIGHT / VALATHU", right_b + 10)
        ]
        
        for text, x_pos in zones:
            cv2.putText(frame, text, (x_pos, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
