import os
import sys
import shutil
import select
import tty
import termios
import subprocess
import tempfile
from pathlib import Path

# Path to your local Piper TTS model (.onnx file)
PIPER_MODEL = os.path.expanduser("~/.local/share/piper/en_US-lessac-medium.onnx")

def get_key(timeout=None):
    """Captures a single keypress immediately without waiting for Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1).lower()
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def play_audio_and_listen(audio_path):
    """
    Plays an audio file via aplay in the background while monitoring for keypresses.
    If a key is pressed, audio stops instantly and the key character is returned.
    """
    if not os.path.exists(audio_path):
        return None

    proc = subprocess.Popen(['aplay', str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    key = None
    while proc.poll() is None:
        key = get_key(timeout=0.05)
        if key is not None:
            proc.terminate()
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
            
    return key

def generate_tts_wav(name):
    """Generates a temporary TTS audio file for the filename."""
    clean_name = Path(name).stem.replace('_', ' ').replace('-', ' ')
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_tts = tf.name
    tf.close()

    try:
        proc = subprocess.Popen(
            ['piper', '--model', PIPER_MODEL, '--output_file', temp_tts],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        proc.communicate(input=clean_name)
        if proc.returncode == 0 and os.path.exists(temp_tts):
            return temp_tts
    except Exception:
        pass

    if os.path.exists(temp_tts):
        os.remove(temp_tts)
    return None

def main():
    cwd = Path.cwd()
    yes_dir = cwd / "yes"
    yes_dir.mkdir(exist_ok=True)

    wav_files = sorted([f for f in cwd.iterdir() if f.is_file() and f.suffix.lower() == '.wav'])

    if not wav_files:
        print("No .wav files found in current directory.")
        return

    print(f"Found {len(wav_files)} .wav file(s).")
    print("Controls (instant single-key, no Enter required):")
    print("  [d] Delete | [a] Add to 'yes/' | [s / Space] Skip | [q] Quit\n")

    try:
        for idx, wav_file in enumerate(wav_files, start=1):
            print(f"[{idx}/{len(wav_files)}] {wav_file.name} ... ", end="", flush=True)

            key = None

            # 1. Play TTS audio name
            tts_path = generate_tts_wav(wav_file.name)
            if tts_path:
                key = play_audio_and_listen(tts_path)
                if os.path.exists(tts_path):
                    os.remove(tts_path)

            # 2. Play original WAV clip (if TTS was not interrupted by keypress)
            if key is None:
                key = play_audio_and_listen(wav_file)

            # 3. Wait indefinitely for user input if audio finished without keypress
            if key is None:
                key = get_key(timeout=None)

            # 4. Immediate Action & Move Next
            if key == 'q':
                print("[Quit]")
                break
            elif key == 'd':
                try:
                    wav_file.unlink()
                    print("[Deleted]")
                except Exception as e:
                    print(f"[Error deleting: {e}]")
            elif key == 'a':
                try:
                    dest = yes_dir / wav_file.name
                    shutil.move(str(wav_file), str(dest))
                    print("[Moved to 'yes/']")
                except Exception as e:
                    print(f"[Error moving: {e}]")
            elif key in ['s', ' ', '\r', '\n']:
                print("[Skipped]")
            else:
                print(f"[Skipped (key '{key}')]")

    except KeyboardInterrupt:
        print("\nSession interrupted.")
    finally:
        print("\nDone processing files.")

if __name__ == '__main__':
    main()
