import threading
import queue
import subprocess
import sys
import json

class AudioEngine(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.queue = queue.Queue()
        self.muted = False
        self.language = "ENGLISH"  # "ENGLISH" or "TAMIL"
        self.tamil_voice_id = None
        self.english_voice_id = None
        self.has_native_tamil = False
        self.lock = threading.Lock()
        self._running = True
        self.proc = None

    def run(self):
        # Spawn the lightweight audio worker subprocess
        # This isolates SAPI5/pyttsx3 inside its own process main-thread, avoiding COM threading failures.
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "audio_worker.py"],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("[AudioEngine] Spawning audio_worker.py subprocess...")
        except Exception as e:
            print(f"[AudioEngine Error] Failed to spawn audio_worker.py: {e}")
            self._running = False
            return

        # Perform handshake to retrieve system voice information
        try:
            ready_line = self.proc.stderr.readline().strip()
            if ready_line.startswith("READY"):
                parts = ready_line.split(":")
                self.has_native_tamil = (parts[1] == "True")
                self.english_voice_id = parts[2]
                self.tamil_voice_id = parts[3]
                print(f"[AudioEngine] Subprocess ready. Native Tamil: {self.has_native_tamil}")
                
                # Start stderr reader thread to drain the pipe and output debug lines
                def log_reader():
                    while self._running and self.proc:
                        try:
                            line = self.proc.stderr.readline()
                            if not line:
                                break
                            print(f"[WorkerLog] {line.strip()}")
                        except Exception:
                            break
                            
                log_thread = threading.Thread(target=log_reader, daemon=True)
                log_thread.start()
            else:
                print(f"[AudioEngine Warning] Unexpected handshake from worker: {ready_line}")
        except Exception as e:
            print(f"[AudioEngine Error] Handshake with worker failed: {e}")
            self._running = False
            return

        # Engine Queue Processing Loop
        while self._running:
            try:
                # Retrieve speaking tasks with timeout
                task = self.queue.get(timeout=0.2)
                if task is None:
                    break

                action = task.get("action")
                text_en = task.get("text_en")
                text_ta_unicode = task.get("text_ta_unicode")
                text_ta_phonetic = task.get("text_ta_phonetic")

                # Get thread-safe configuration details
                with self.lock:
                    is_muted = self.muted
                    current_lang = self.language

                # Formulate JSON command payload
                payload = {
                    "action": action,
                    "text_en": text_en,
                    "text_ta_unicode": text_ta_unicode,
                    "text_ta_phonetic": text_ta_phonetic,
                    "language": current_lang,
                    "muted": is_muted
                }

                # Send command to isolated worker process
                if self.proc and self.proc.poll() is None:
                    try:
                        self.proc.stdin.write(json.dumps(payload) + "\n")
                        self.proc.stdin.flush()
                    except Exception as e:
                        print(f"[AudioEngine Error] Failed to write to worker process: {e}")
                else:
                    print("[AudioEngine Error] Audio worker process is not running.")

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioEngine Loop Exception]: {e}")

        # Shutdown subprocess cleanly
        self._terminate_subprocess()

    def _terminate_subprocess(self):
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.stdin.write(json.dumps({"action": "stop"}) + "\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def speak(self, text_en, text_ta_unicode, text_ta_phonetic):
        """Queue up a spatial obstacle speech task."""
        if self._running:
            self.queue.put({
                "action": "speak",
                "text_en": text_en,
                "text_ta_unicode": text_ta_unicode,
                "text_ta_phonetic": text_ta_phonetic
            })

    def announce(self, text_en, text_ta_unicode, text_ta_phonetic):
        """Queue up a system announcement (ignoring standard mute state)."""
        if self._running:
            self.queue.put({
                "action": "announce",
                "text_en": text_en,
                "text_ta_unicode": text_ta_unicode,
                "text_ta_phonetic": text_ta_phonetic
            })

    def set_muted(self, muted):
        with self.lock:
            self.muted = muted

    def get_muted(self):
        with self.lock:
            return self.muted

    def set_language(self, language):
        with self.lock:
            if language in ["ENGLISH", "TAMIL"]:
                self.language = language

    def get_language(self):
        with self.lock:
            return self.language

    def stop(self):
        self._running = False
        self.queue.put(None)
        # Give thread time to terminate subprocess inside run()
