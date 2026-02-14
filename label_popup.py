import sqlite3
import time
import tkinter as tk

def submit_stress(level):
    conn = sqlite3.connect("keystrokes.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stress_labels (timestamp, stress_level) VALUES (?, ?)",
        (time.time(), level)
    )
    conn.commit()
    conn.close()
    root.destroy()

root = tk.Tk()
root.title("Stress Level")

tk.Label(root, text="How stressed are you right now? (1–5)").pack(pady=10)

for i in range(1, 6):
    tk.Button(root, text=str(i), width=10,
              command=lambda level=i: submit_stress(level)).pack(pady=2)

root.mainloop()
