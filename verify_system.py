import time
import numpy as np
from audio_engine import AudioEngine
from vision_processor import VisionProcessor

def verify():
    print("----- STARTING SYSTEM VERIFICATION -----")
    
    # 1. Test AudioEngine init
    print("Testing AudioEngine initialization...")
    engine = AudioEngine()
    engine.start()
    time.sleep(1)
    
    # Check if voice was configured
    print(f" - Native Tamil voice available: {engine.has_native_tamil}")
    print(f" - Designated English voice ID: {engine.english_voice_id}")
    print(f" - Designated Tamil voice ID: {engine.tamil_voice_id}")
    
    # Test queue speech (non-blocking)
    print("Enqueuing test announcement...")
    engine.announce(
        "Verification sequence initiated.",
        "சரிபார்ப்பு வரிசை தொடங்கப்பட்டது.",
        "Saribaarppu varisai thodangappattathu."
    )
    time.sleep(2)
    
    # 2. Test VisionProcessor init (This will download yolov8n.pt if not present)
    print("Testing VisionProcessor initialization & model loading...")
    processor = VisionProcessor(model_name="yolov8n.pt")
    print("VisionProcessor loaded successfully.")
    
    # 3. Process a dummy frame
    print("Testing YOLOv8 frame processing on a mock image...")
    # Create black frame with three colored squares to mimic objects
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Let's draw some text to run through the model
    # (YOLOv8n might not detect simple drawings, but we can verify it doesn't crash)
    processed_frame, detections = processor.process_frame(frame)
    print(f" - Frame processed successfully. Detections count: {len(detections)}")
    
    # 4. Test bilingual translation logic
    print("Testing bilingual translation logic...")
    mock_detection = {
        "label": "chair",
        "confidence": 0.85,
        "bbox": (100, 100, 300, 400),
        "zone": "Left",
        "proximity": "Close"
    }
    en_phrase, ta_unicode, ta_phonetic = processor.format_voice_cues(mock_detection)
    print(f" - English phrase: '{en_phrase}'")
    print(f" - Tamil unicode: '{ta_unicode}'")
    print(f" - Tamil phonetic: '{ta_phonetic}'")
    
    # Verify expected outputs
    assert en_phrase == "A chair is close to you, on your left side.", f"Unexpected English: {en_phrase}"
    assert ta_unicode == "நாற்காலி உங்களுக்கு இடது பக்கத்தில் அருகில் உள்ளது.", f"Unexpected Tamil unicode: {ta_unicode}"
    assert ta_phonetic == "Naarkali ungalukku idathu pakkathil arugil ullathu.", f"Unexpected Tamil phonetic: {ta_phonetic}"
    print(" - Translation phrases verified successfully!")

    # Enqueue a speaking command with the verified phrases
    print("Enqueuing test speech for mock detection...")
    engine.speak(en_phrase, ta_unicode, ta_phonetic)
    time.sleep(2)
    
    # Shutdown audio engine
    print("Stopping AudioEngine...")
    engine.stop()
    engine.join(timeout=2.0)
    print("----- SYSTEM VERIFICATION COMPLETED SUCCESSFULLY -----")

if __name__ == "__main__":
    verify()
