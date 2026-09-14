#!/usr/bin/env bash

ROOT="$(pwd)"
DEVICE="Speaker group all"
INTERVAL=180
REPEATS=3
DB="$ROOT/.played_wavs.json"

export PATH="$HOME/.local/bin:$PATH"

# ------------------------------------------------------------
# Install catt automatically if needed
# ------------------------------------------------------------

if ! command -v catt >/dev/null 2>&1; then
    echo "catt is not installed. Installing..."

    if ! command -v pipx >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y pipx
        else
            python3 -m pip install --user pipx
        fi
    fi

    export PATH="$HOME/.local/bin:$PATH"

    pipx install catt || true
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v catt >/dev/null 2>&1; then
    echo "ERROR: catt installation failed."
    exit 1
fi

CATT="$(command -v catt)"

# ------------------------------------------------------------
# Create JSON database if it doesn't exist
# ------------------------------------------------------------

if [[ ! -f "$DB" ]]; then
    echo '{}' > "$DB"
fi

# Make sure database is valid JSON.
if ! python3 -m json.tool "$DB" >/dev/null 2>&1; then
    echo "ERROR: $DB contains invalid JSON."
    exit 1
fi

echo "========================================"
echo " WAV Chromecast Watcher"
echo "========================================"
echo
echo "Directory: $ROOT"
echo "Device:    $DEVICE"
echo "Database:  $DB"
echo "Interval:  $INTERVAL seconds"
echo "Repeats:   $REPEATS"
echo

# ------------------------------------------------------------
# Check whether file is already in database
# ------------------------------------------------------------

already_played() {
    local file="$1"

    python3 - "$DB" "$file" <<'PY'
import json
import sys

dbfile = sys.argv[1]
filename = sys.argv[2]

with open(dbfile, "r", encoding="utf-8") as f:
    db = json.load(f)

sys.exit(0 if filename in db else 1)
PY
}

# ------------------------------------------------------------
# Add successfully played file to database
# ------------------------------------------------------------

mark_played() {
    local file="$1"

    python3 - "$DB" "$file" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime

dbfile = sys.argv[1]
filename = sys.argv[2]

with open(dbfile, "r", encoding="utf-8") as f:
    db = json.load(f)

try:
    st = os.stat(filename)
    size = st.st_size
    mtime = st.st_mtime
except OSError:
    size = None
    mtime = None

db[filename] = {
    "played_at": datetime.now().astimezone().isoformat(),
    "size": size,
    "mtime": mtime
}

directory = os.path.dirname(dbfile) or "."

fd, tmp = tempfile.mkstemp(
    prefix=".played_wavs.",
    suffix=".tmp",
    dir=directory
)

try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, sort_keys=True)
        f.write("\n")

    os.replace(tmp, dbfile)

except:
    try:
        os.unlink(tmp)
    except:
        pass
    raise
PY
}

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

while true; do

    while IFS= read -r -d '' file; do

        # Get canonical absolute path.
        file="$(realpath "$file")"

        # Skip anything that's already been successfully played.
        if already_played "$file"; then
            continue
        fi

        echo
        echo "========================================"
        echo "Unplayed WAV found:"
        echo "$file"
        echo "========================================"

        # Make sure the WAV has stopped changing before playing it.
        size1=$(stat -c %s "$file" 2>/dev/null) || continue
        sleep 5
        size2=$(stat -c %s "$file" 2>/dev/null) || continue

        if [[ "$size1" != "$size2" ]]; then
            echo "File is still being written. Skipping until next scan."
            continue
        fi

        success=1

        for ((i=1; i<=REPEATS; i++)); do
            echo "Playing $i of $REPEATS..."

            if ! "$CATT" -d "$DEVICE" cast "$file"; then
                echo "Cast failed."
                success=0
                break
            fi
        done

        if (( success )); then
            mark_played "$file"
            echo "Finished and recorded in database."
        else
            echo "Not added to database; will retry later."
        fi

    done < <(
        find "$ROOT" \
            -type f \
            -iname '*.wav' \
            -print0
    )

    echo
    echo "Scan complete. Next scan in $INTERVAL seconds..."
    sleep "$INTERVAL"

done
