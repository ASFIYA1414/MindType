import sqlite3
import pandas as pd
import numpy as np

WINDOW_SIZE = 30 # seconds

conn = sqlite3.connect("keystrokes.db")

# Load tables
keystrokes = pd.read_sql_query("SELECT * FROM keystrokes", conn)
stress_labels = pd.read_sql_query("SELECT * FROM stress_labels", conn)

conn.close()

print("Total keystrokes:", len(keystrokes))
print("Total stress labels:", len(stress_labels))

# Sort by timestamp
keystrokes = keystrokes.sort_values("timestamp")

dataset = []

# Group by user and session
grouped = keystrokes.groupby(["user_id", "session_id"])

for (user, session), group in grouped:

    session_start = group["timestamp"].min()
    session_end = group["timestamp"].max()

    current_start = session_start

    while current_start + WINDOW_SIZE <= session_end:
        current_end = current_start + WINDOW_SIZE

        window = group[
            (group["timestamp"] >= current_start) &
            (group["timestamp"] < current_end)
        ]

        if len(window) < 10:
            current_start += WINDOW_SIZE
            continue

        # Compute holds
        presses = window[window["event_type"] == "press"]
        releases = window[window["event_type"] == "release"]

        holds = []

        for _, press in presses.iterrows():
            key = press["key"]
            release_match = releases[
                (releases["key"] == key) &
                (releases["timestamp"] > press["timestamp"])
            ]

            if not release_match.empty:
                release_time = release_match.iloc[0]["timestamp"]
                holds.append(release_time - press["timestamp"])

        if len(holds) == 0:
            current_start += WINDOW_SIZE
            continue

        avg_hold = np.mean(holds)
        hold_variance = np.var(holds)

        # Pauses between key presses
        press_times = presses["timestamp"].values
        pauses = np.diff(press_times)

        avg_pause = np.mean(pauses) if len(pauses) > 0 else 0

        # KPM (keys per minute)
        kpm = len(presses)

        # Backspace rate
        backspaces = presses[presses["key"] == "Key.backspace"]
        backspace_rate = len(backspaces) / len(presses)

        # Find nearest stress label
        labels = stress_labels[
            (stress_labels["user_id"] == user) &
            (stress_labels["session_id"] == session)
        ]

        if labels.empty:
            current_start += WINDOW_SIZE
            continue

        labels = labels.copy()
        labels.loc[:, "time_diff"] = abs(labels["timestamp"] - current_start)
        nearest_label = labels.sort_values("time_diff").iloc[0]["stress_level"]

        dataset.append([
            user,
            session,
            avg_hold,
            hold_variance,
            avg_pause,
            kpm,
            backspace_rate,
            nearest_label
        ])

        current_start += WINDOW_SIZE

columns = [
    "user_id",
    "session_id",
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate",
    "stress_level"
]

final_df = pd.DataFrame(dataset, columns=columns)

final_df.to_csv("mindtype_dataset.csv", index=False)

print("\nDataset created successfully ✅")
print("Total samples:", len(final_df))
