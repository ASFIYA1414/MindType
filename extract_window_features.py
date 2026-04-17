import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import entropy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

log.info("\n" + "="*70)
log.info("EXTRACTING KEYSTROKE DYNAMICS FEATURES (STRESS v2)")
log.info("="*70)

WINDOW_SIZE = 30  # seconds

conn = sqlite3.connect("keystrokes.db")

# Load tables
keystrokes = pd.read_sql_query("SELECT * FROM keystrokes", conn)
stress_labels = pd.read_sql_query("SELECT * FROM stress_labels", conn)

conn.close()

log.info(f"\nTotal keystrokes: {len(keystrokes)}")
log.info(f"Total stress labels: {len(stress_labels)}")

# Sort by timestamp
keystrokes = keystrokes.sort_values("timestamp")

# Pre-compute user-level KPM statistics for session z-score normalization
log.info("\nPre-computing user-level KPM statistics...")
user_kpm_stats = {}
for user_id in keystrokes["user_id"].unique():
    user_keystrokes = keystrokes[keystrokes["user_id"] == user_id]
    presses = user_keystrokes[user_keystrokes["event_type"] == "press"]
    total_keys = len(presses)
    
    # Average KPM across all windows for this user
    if total_keys > 0:
        user_kpm_stats[user_id] = {
            "mean": total_keys / 60,
            "std": 1.0  # Will be refined per user
        }

dataset = []

# Group by user and session
grouped = keystrokes.groupby(["user_id", "session_id"])

log.info(f"\nExtracting features from {len(grouped)} user-session groups...")

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

        # =========================================
        # ORIGINAL 5 FEATURES
        # =========================================
        
        avg_hold = np.mean(holds)
        hold_variance = np.var(holds)

        # Pauses between key presses (inter-key intervals)
        press_times = presses["timestamp"].values
        pauses = np.diff(press_times)

        avg_pause = np.mean(pauses) if len(pauses) > 0 else 0

        # KPM (keys per minute)
        kpm = len(presses)

        # Backspace rate
        backspaces = presses[presses["key"] == "Key.backspace"]
        backspace_rate = len(backspaces) / max(len(presses), 1)

        # =========================================
        # NEW FEATURES v2 (HIGH-SIGNAL FOR STRESS)
        # =========================================
        
        # 1. IKI ENTROPY (Inter-Key Interval Entropy)
        # Stressed typists have irregular pauses; entropy captures unpredictability
        if len(pauses) > 1:
            hist_counts, _ = np.histogram(pauses, bins=10)
            hist_counts = hist_counts[hist_counts > 0]  # Remove zero bins
            iki_entropy = entropy(hist_counts) if len(hist_counts) > 0 else 0
        else:
            iki_entropy = 0

        # 2. HOLD-TIME COEFFICIENT OF VARIATION (Hold CV)
        # Normalizes hold variance for typing speed; pure variance misleads
        hold_mean = np.mean(holds) if len(holds) > 0 else 1e-8
        hold_cv = (np.std(holds) / hold_mean) if hold_mean > 0 else 0

        # 3. DIGRAPH LATENCY (Key transition time)
        # Transition time between common bigrams degrades under cognitive load
        # For simplicity, compute mean latency for space-preceded keys
        digraph_latencies = []
        for i in range(len(press_times) - 1):
            latency = press_times[i + 1] - press_times[i]
            if latency > 0.01:  # Filter out simultaneous presses
                digraph_latencies.append(latency)
        
        digraph_latency = np.mean(digraph_latencies) if len(digraph_latencies) > 0 else 0

        # 4. ERROR BURST RATE
        # Stress causes clusters of corrections, not isolated typos
        backspace_times = presses[presses["key"] == "Key.backspace"]["timestamp"].values
        error_bursts = 0
        if len(backspace_times) > 1:
            inter_error_intervals = np.diff(backspace_times)
            # Burst = 2+ backspaces within 0.5 seconds
            burst_threshold = 0.5
            consecutive_errors = 1
            for interval in inter_error_intervals:
                if interval <= burst_threshold:
                    consecutive_errors += 1
                else:
                    if consecutive_errors >= 2:
                        error_bursts += 1
                    consecutive_errors = 1
            if consecutive_errors >= 2:
                error_bursts += 1
        
        error_burst_rate = error_bursts / max(len(backspaces), 1) if len(backspaces) > 0 else 0

        # 5. SESSION KPM Z-SCORE
        # Relative speed drop is more meaningful than absolute KPM
        user_kpm_mean = user_kpm_stats.get(user, {}).get("mean", 1)
        user_kpm_std = max(user_kpm_stats.get(user, {}).get("std", 1), 1e-6)
        session_kpm_z = (kpm - user_kpm_mean) / user_kpm_std if user_kpm_std > 0 else 0

        # =========================================
        # Find nearest stress label
        # =========================================
        
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

        # =========================================
        # APPEND COMPLETE FEATURE VECTOR
        # =========================================
        
        dataset.append([
            user,
            session,
            avg_hold,
            hold_variance,
            avg_pause,
            kpm,
            backspace_rate,
            iki_entropy,           # NEW
            hold_cv,               # NEW
            digraph_latency,       # NEW
            error_burst_rate,      # NEW
            session_kpm_z,         # NEW
            nearest_label
        ])

        current_start += WINDOW_SIZE

# =========================================
# CREATE DATAFRAME & SAVE
# =========================================

columns = [
    "user_id",
    "session_id",
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate",
    "iki_entropy",           # NEW
    "hold_cv",               # NEW
    "digraph_latency",       # NEW
    "error_burst_rate",      # NEW
    "session_kpm_z",         # NEW
    "stress_level"
]

final_df = pd.DataFrame(dataset, columns=columns)

log.info(f"\n" + "="*70)
log.info("FEATURE EXTRACTION COMPLETE")
log.info("="*70)
log.info(f"Total samples: {len(final_df)}")
log.info(f"Features: {len(columns) - 3} (user_id, session_id excluded)")

# Validate extracted features
log.info(f"\n--- FEATURE STATISTICS ---")
feature_cols = [c for c in columns if c not in ["user_id", "session_id", "stress_level"]]
for col in feature_cols:
    log.info(f"{col:20s}: mean={final_df[col].mean():10.4f}, std={final_df[col].std():10.4f}, "
             f"min={final_df[col].min():10.4f}, max={final_df[col].max():10.4f}")

log.info(f"\n--- CLASS DISTRIBUTION ---")
log.info(f"\nStress level distribution:")
log.info(final_df["stress_level"].value_counts().sort_index())

final_df.to_csv("mindtype_dataset.csv", index=False)

log.info(f"\n✅ Dataset saved: mindtype_dataset.csv")
log.info("="*70 + "\n")
