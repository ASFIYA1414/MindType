from pynput import keyboard
import time
import sqlite3

conn = sqlite3.connect("keystrokes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS keystrokes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT,
    event_type TEXT,
    timestamp REAL
)
""")

# Create stress labels table
cursor.execute("""
CREATE TABLE IF NOT EXISTS stress_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    stress_level INTEGER
)
""")

conn.commit()

# ---------------- KEY EVENT FUNCTIONS ----------------
def on_press(key):
    timestamp = time.time()

    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)

    cursor.execute(
        "INSERT INTO keystrokes (key, event_type, timestamp) VALUES (?, ?, ?)",
        (key_name, "press", timestamp)
    )
    conn.commit()


def on_release(key):
    timestamp = time.time()

    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)

    cursor.execute(
        "INSERT INTO keystrokes (key, event_type, timestamp) VALUES (?, ?, ?)",
        (key_name, "release", timestamp)
    )
    conn.commit()

    if key == keyboard.Key.esc:
        print("Stopping capture...")
        conn.close()
        return False


print("Recording keystrokes... Press ESC to stop.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()