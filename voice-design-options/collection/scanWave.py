from pathlib import Path

def main():
    cwd = Path.cwd().resolve()
    
    # Get all filenames currently present in ./
    existing_files = {f.name for f in cwd.iterdir() if f.is_file()}
    
    missing_wavs = []

    # Search recursively in parent directories (..)
    parent_dir = Path('..').resolve()
    for wav_path in parent_dir.rglob('*.wav'):
        wav_resolved = wav_path.resolve()
        
        # Skip files that reside inside the current directory to avoid false positives
        try:
            wav_resolved.relative_to(cwd)
            continue
        except ValueError:
            pass
            
        # Check if filename is missing from ./
        if wav_path.name not in existing_files:
            missing_wavs.append(str(wav_path))

    print(f"Found {len(missing_wavs)} missing .wav file(s):\n")
    for fp in missing_wavs:
        print(fp)

if __name__ == '__main__':
    main()
