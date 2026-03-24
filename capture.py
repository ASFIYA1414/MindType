from pynput import keyboard
import time
import sqlite3
import tkinter as tk
import threading

# ---------------- USER INPUT ----------------
user_id = input("Enter User ID (e.g., U01): ")
condition = input("Enter Condition (calm/stress): ")
session_id = int(time.time())  # unique session ID

print("Session ID:", session_id)

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect("keystrokes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS keystrokes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    session_id INTEGER,
    condition TEXT,
    key TEXT,
    event_type TEXT,
    timestamp REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS stress_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    session_id INTEGER,
    timestamp REAL,
    stress_level INTEGER
)
""")

conn.commit()

# ---------------- STRESS POPUP ----------------
def submit_stress(level):
    conn2 = sqlite3.connect("keystrokes.db")
    cursor2 = conn2.cursor()

    cursor2.execute("""
        INSERT INTO stress_labels (user_id, session_id, timestamp, stress_level)
        VALUES (?, ?, ?, ?)
    """, (user_id, session_id, time.time(), level))

    conn2.commit()
    conn2.close()

    popup.destroy()


def show_popup():
    global popup
    popup = tk.Tk()
    popup.title("Stress Level")

    tk.Label(popup, text="Current Stress Level").pack(pady=10)

    tk.Button(popup, text="Calm", width=15,
              command=lambda: submit_stress(0)).pack(pady=5)

    tk.Button(popup, text="Moderate", width=15,
              command=lambda: submit_stress(1)).pack(pady=5)

    tk.Button(popup, text="High", width=15,
              command=lambda: submit_stress(2)).pack(pady=5)

    popup.mainloop()


def popup_loop():
    while True:
        time.sleep(30)
        show_popup()


# Start popup thread
popup_thread = threading.Thread(target=popup_loop)
popup_thread.daemon = True
popup_thread.start()

# ---------------- KEY EVENT FUNCTIONS ----------------
def on_press(key):
    timestamp = time.time()

    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)

    cursor.execute(
        "INSERT INTO keystrokes (user_id, session_id, condition, key, event_type, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, session_id, condition, key_name, "press", timestamp)
    )
    conn.commit()


def on_release(key):
    timestamp = time.time()

    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)

    cursor.execute(
        "INSERT INTO keystrokes (user_id, session_id, condition, key, event_type, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, session_id, condition, key_name, "release", timestamp)
    )
    conn.commit()

    if key == keyboard.Key.esc:
        print("Stopping capture...")
        conn.close()
        return False


print("Recording keystrokes... Press ESC to stop.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
