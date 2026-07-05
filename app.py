import cv2
import time
import sys
from audio_engine import AudioEngine
from vision_processor import VisionProcessor

def main():
    print("================================================================")
    print("       BILINGUAL SMART VISION ASSIST SYSTEM FOR VISUALLY IMPAIRED")
    print("================================================================")
    print("[Initialization] Setting up system modules...")

    # 1. Initialize the thread-safe Audio Engine
    audio_engine = AudioEngine()
    audio_engine.start()

    # 2. Initialize the Vision Processor (YOLOv8 and translations)
    try:
        processor = VisionProcessor(model_name="yolov8n.pt")
        print("[Initialization] YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"[Initialization Error] Failed to load YOLOv8 model: {e}")
        audio_engine.stop()
        sys.exit(1)

    # 3. Setup Video Capture (Fallback to Virtual Simulation if Webcam is not available)
    cap = cv2.VideoCapture(0)
    use_simulator = False
    
    if not cap.isOpened():
        print("\n[Webcam Warning] Physical camera (index 0) could not be opened.")
        print("[Webcam Warning] Swapping to VIRTUAL HARDWARE STREAM SIMULATOR for demonstration.")
        use_simulator = True
    else:
        # Set parameters for smooth capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Attempt to request 30 FPS fallback
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("[Webcam] Successfully initialized camera capture at 640x480 resolution.")

    # 4. Audio Control and Cooldown Variables
    cooldown_duration = 3.5  # Throttle: 3 to 4 seconds
    last_speech_time = 0.0
    last_spoken_target = None
    frame_idx = 0  # Used for simulated animation

    # Immediate announcement on system launch
    audio_engine.announce(
        "System initiated. Language set to English.",
        "கணினி தொடங்கப்பட்டது. மொழி தமிழ்.",
        "Kani-ni thodangappattathu. Mozhi Tamil."
    )

    print("\n[Controls] Active Keyboard Simulation Hotkeys:")
    print(" - Press [SPACEBAR] to Toggle Audio Assistance [RUNNING / MUTED]")
    print(" - Press [L] to Swap Speech Language [ENGLISH / TAMIL]")
    print(" - Press [ESC] to Safely Shutdown and Exit Application\n")

    while True:
        start_time = time.time()
        
        # 5. Retrieve frame
        if use_simulator:
            # Generate dummy frame
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Create a nice dark gray grid matrix to mimic camera pixels
            for i in range(0, 640, 40):
                cv2.line(frame, (i, 0), (i, 480), (25, 25, 25), 1)
            for i in range(0, 480, 40):
                cv2.line(frame, (0, i), (640, i), (25, 25, 25), 1)
                
            cv2.putText(frame, "* VIRTUAL CAMERA SIMULATOR *", (190, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
            
            # Simulate a moving object ("chair" or "backpack" or "person")
            frame_idx += 1
            # Move object horizontally using sine wave
            mid_x_sim = int(320 + 220 * np.sin(frame_idx * 0.015))
            # Oscilate height to trigger proximity changes
            box_h_sim = int(220 + 90 * np.cos(frame_idx * 0.025))
            box_w_sim = int(box_h_sim * 0.65)
            
            x1 = max(0, mid_x_sim - box_w_sim // 2)
            y1 = max(90, 240 - box_h_sim // 2)
            x2 = min(640, mid_x_sim + box_w_sim // 2)
            y2 = min(450, 240 + box_h_sim // 2)
            
            # Determine zone and proximity parameters
            left_b = 640 // 3
            right_b = (2 * 640) // 3
            zone = "Left" if mid_x_sim < left_b else ("Center" if mid_x_sim < right_b else "Right")
            
            height_ratio = box_h_sim / 480
            proximity = "Close" if height_ratio >= 0.45 else "Far"
            
            # Make the dummy frame draw a cyan visual target wireframe representation
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (mid_x_sim, y1 + box_h_sim // 4), box_w_sim // 4, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (mid_x_sim, y1 + box_h_sim // 2), (mid_x_sim, y2 - 20), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "SIMULATING TARGET: chair", (x1 + 5, y1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Formulate simulated detections
            detections = [{
                "label": "chair",
                "confidence": 0.88,
                "bbox": (x1, y1, x2, y2),
                "zone": zone,
                "proximity": proximity
            }]
            
            # Draw standard grid divisions and line guides
            processor.draw_translucent_zone_lines(frame, left_b, right_b, 480, 640)
            
            # Draw overlay features on top of simulation
            for det in detections:
                tx1, ty1, tx2, ty2 = det["bbox"]
                tcolor = (46, 204, 113) if det["proximity"] == "Close" else (52, 152, 219)
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), tcolor, 2, cv2.LINE_AA)
                processor.draw_corner_brackets(frame, tx1, ty1, tx2, ty2, tcolor)
                
                metric_text = f"{det['label'].capitalize()} ({int(det['confidence']*100)}%) - {det['zone']} | {det['proximity']}"
                text_size = cv2.getTextSize(metric_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                processor.draw_translucent_rect(frame, tx1, max(ty1 - 22, 0), min(tx1 + text_size[0] + 12, 640), ty1, tcolor, alpha=0.7)
                cv2.putText(frame, metric_text, (tx1 + 6, ty1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
            time.sleep(0.03)  # Maintain roughly 30 FPS stream simulation
        else:
            ret, raw_frame = cap.read()
            if not ret:
                print("[Webcam Error] Failed to grab frame from video device.")
                break
            
            # Process using YOLO and get annotations
            frame, detections = processor.process_frame(raw_frame)

        # 6. Audio Throttling Logic
        # Speak the highest-confidence object (the first sorted detection) when cooldown timer permits
        current_time = time.time()
        is_muted = audio_engine.get_muted()
        
        if detections:
            highest_conf_obj = detections[0]
            time_diff = current_time - last_speech_time
            
            # Debug log to trace why it's not speaking
            print(f"[App Debug] Detected: {highest_conf_obj['label']} (conf: {highest_conf_obj['confidence']:.2f}). Muted: {is_muted}. Cooldown diff: {time_diff:.1f}s / {cooldown_duration}s")
            
            if not is_muted and time_diff >= cooldown_duration:
                # Generate bilingual phrasing
                en_phrase, ta_unicode, ta_phonetic = processor.format_voice_cues(highest_conf_obj)
                
                print(f"[App -> Speech Request] Enqueuing: '{en_phrase}'")
                
                # Request background speech task (non-blocking)
                audio_engine.speak(en_phrase, ta_unicode, ta_phonetic)
                
                # Update tracking timers
                last_speech_time = current_time
                last_spoken_target = (en_phrase, ta_unicode, ta_phonetic)

        # 7. Render Premium HUD Overlay Graphics (Top HUD panel + Audio metrics panel)
        h, w, _ = frame.shape
        
        # Draw translucent glassmorphism background panels
        processor.draw_translucent_rect(frame, 10, 10, 270, 105, (15, 15, 15), alpha=0.85)
        cv2.rectangle(frame, (10, 10), (270, 105), (100, 100, 100), 1, lineType=cv2.LINE_AA)

        processor.draw_translucent_rect(frame, w - 320, 10, w - 10, 105, (15, 15, 15), alpha=0.85)
        cv2.rectangle(frame, (w - 320, 10), (w - 10, 105), (100, 100, 100), 1, lineType=cv2.LINE_AA)

        # Draw HUD - System Status (RUNNING / MUTED)
        status_text = "MUTED" if is_muted else "RUNNING"
        status_color = (46, 204, 113) if not is_muted else (231, 76, 60) # Green vs Red
        cv2.putText(frame, "SYSTEM STATUS:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (135, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2, cv2.LINE_AA)

        # Draw HUD - Active Language (ENGLISH / TAMIL)
        active_lang = audio_engine.get_language()
        lang_color = (241, 196, 15) if active_lang == "TAMIL" else (52, 152, 219) # Gold vs Blue
        cv2.putText(frame, "ACTIVE LANG :", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, active_lang, (135, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lang_color, 2, cv2.LINE_AA)

        # Draw HUD - Keyboard shortcuts guide
        cv2.putText(frame, "[SPACEBAR] Audio On/Off", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, "[L KEY]    Swap Language", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

        # Draw Right HUD - Audio Target Tracking info
        cv2.putText(frame, "AUDIO FEEDBACK TARGET:", (w - 310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        if last_spoken_target:
            current_lang = audio_engine.get_language()
            phrase_display = last_spoken_target[2] if current_lang == "TAMIL" else last_spoken_target[0]
            # Use smaller font scale and wrap/clip nicely
            if len(phrase_display) > 52:
                phrase_display = phrase_display[:49] + "..."
                
            cv2.putText(frame, f"\"{phrase_display}\"", (w - 310, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (241, 196, 15), 1, cv2.LINE_AA)
            
            # Cooldown dynamic progress bar
            time_elapsed = current_time - last_speech_time
            cooldown_left = max(0.0, cooldown_duration - time_elapsed)
            bar_percentage = min(1.0, time_elapsed / cooldown_duration)
            
            # Draw outline bar
            cv2.rectangle(frame, (w - 310, 68), (w - 20, 76), (60, 60, 60), -1)
            # Draw progress bar fill
            bar_fill_w = int(290 * bar_percentage)
            bar_fill_color = (46, 204, 113) if cooldown_left == 0.0 else (230, 126, 34)
            cv2.rectangle(frame, (w - 310, 68), (w - 310 + bar_fill_w, 76), bar_fill_color, -1)
            
            cooldown_status = "READY TO VOICE" if cooldown_left == 0.0 else f"COOLDOWN: {cooldown_left:.1f}s"
            cv2.putText(frame, cooldown_status, (w - 310, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No descriptions spoken yet.", (w - 310, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(frame, "READY TO VOICE", (w - 310, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (46, 204, 113), 1, cv2.LINE_AA)

        # Show Output Frame in OpenCV GUI Window
        cv2.imshow("Bilingual Smart Vision Assist System", frame)

        # 8. Keyboard Simulation Inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Spacebar Toggle Muted state
        if key == 32:  
            new_mute = not audio_engine.get_muted()
            audio_engine.set_muted(new_mute)
            if new_mute:
                print("[Action] Audio Muted.")
                audio_engine.announce(
                    "Voice assistance deactivated.",
                    "ஒலி உதவி முடக்கப்பட்டது.",
                    "Oli uthavi mudakkappattathu."
                )
            else:
                print("[Action] Audio Unmuted.")
                audio_engine.announce(
                    "Voice assistance activated.",
                    "ஒலி உதவி செயல்படுத்தப்பட்டது.",
                    "Oli uthavi seyalpaduthappattathu."
                )
                # Reset cooldown immediately to notify on next frame
                last_speech_time = 0.0
                
        # 'L' Key Swap language
        elif key == ord('l') or key == ord('L'):
            current_lang = audio_engine.get_language()
            new_lang = "TAMIL" if current_lang == "ENGLISH" else "ENGLISH"
            audio_engine.set_language(new_lang)
            print(f"[Action] Language toggled to: {new_lang}")
            
            # Instantly cue the language change in the new language
            if new_lang == "ENGLISH":
                audio_engine.announce(
                    "Language set to English.",
                    "மொழி ஆங்கிலத்திற்கு மாற்றப்பட்டது.",
                    "Mozhi Aangilathirkku maatrapattathu."
                )
            else:
                audio_engine.announce(
                    "Language set to Tamil.",
                    "மொழி தமிழுக்கா மாற்றப்பட்டது.",
                    "Mozhi Tamilukka maatrapattathu."
                )
            # Reset cooldown so user hears first detection in new language immediately
            last_speech_time = 0.0

        # ESC Key to Safely Quit
        elif key == 27:
            print("[Action] Shutdown request received. Exiting...")
            break

    # Clean Exit and System Resource Release
    print("[Shutdown] Closing video capture...")
    if not use_simulator:
        cap.release()
    print("[Shutdown] Terminating background audio engine thread...")
    audio_engine.stop()
    audio_engine.join(timeout=2.0)
    cv2.destroyAllWindows()
    print("[Shutdown] Smart Vision Assist System terminated cleanly.")

if __name__ == "__main__":
    main()
