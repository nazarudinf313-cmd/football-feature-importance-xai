"""
train_model.py
Trains the Random Forest classifier used in this study and saves
the fitted model + feature list to model.pkl, so app.py (Streamlit)
can load it without re-running the full notebook.

Run this AFTER placing database.sqlite inside the data/ folder:
    python train_model.py
"""

import sqlite3
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DB_PATH = "data/database.sqlite"
MODEL_OUT = "model.pkl"

FEATURES = [
    "home_buildUpPlaySpeed", "home_buildUpPlayDribbling", "home_buildUpPlayPassing",
    "home_chanceCreationPassing", "home_chanceCreationCrossing", "home_chanceCreationShooting",
    "home_defencePressure", "home_defenceAggression", "home_defenceTeamWidth",
    "away_buildUpPlaySpeed", "away_buildUpPlayDribbling", "away_buildUpPlayPassing",
    "away_chanceCreationPassing", "away_chanceCreationCrossing", "away_chanceCreationShooting",
    "away_defencePressure", "away_defenceAggression", "away_defenceTeamWidth",
]

QUERY = """
SELECT
    m.home_team_goal, m.away_team_goal,
    hta.buildUpPlaySpeed AS home_buildUpPlaySpeed,
    hta.buildUpPlayDribbling AS home_buildUpPlayDribbling,
    hta.buildUpPlayPassing AS home_buildUpPlayPassing,
    hta.chanceCreationPassing AS home_chanceCreationPassing,
    hta.chanceCreationCrossing AS home_chanceCreationCrossing,
    hta.chanceCreationShooting AS home_chanceCreationShooting,
    hta.defencePressure AS home_defencePressure,
    hta.defenceAggression AS home_defenceAggression,
    hta.defenceTeamWidth AS home_defenceTeamWidth,
    ata.buildUpPlaySpeed AS away_buildUpPlaySpeed,
    ata.buildUpPlayDribbling AS away_buildUpPlayDribbling,
    ata.buildUpPlayPassing AS away_buildUpPlayPassing,
    ata.chanceCreationPassing AS away_chanceCreationPassing,
    ata.chanceCreationCrossing AS away_chanceCreationCrossing,
    ata.chanceCreationShooting AS away_chanceCreationShooting,
    ata.defencePressure AS away_defencePressure,
    ata.defenceAggression AS away_defenceAggression,
    ata.defenceTeamWidth AS away_defenceTeamWidth
FROM Match m
JOIN Team_Attributes hta
    ON hta.id = (
        SELECT id FROM Team_Attributes
        WHERE team_api_id = m.home_team_api_id AND date <= m.date
        ORDER BY date DESC LIMIT 1
    )
JOIN Team_Attributes ata
    ON ata.id = (
        SELECT id FROM Team_Attributes
        WHERE team_api_id = m.away_team_api_id AND date <= m.date
        ORDER BY date DESC LIMIT 1
    )
WHERE m.home_team_goal IS NOT NULL AND m.away_team_goal IS NOT NULL
"""


def get_result(row):
    if row["home_team_goal"] > row["away_team_goal"]:
        return 2  # Home Win
    elif row["home_team_goal"] < row["away_team_goal"]:
        return 0  # Away Win
    else:
        return 1  # Draw


def main():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(QUERY, conn)
    conn.close()
    print(f"Loaded {len(df)} matches.")

    df["result"] = df.apply(get_result, axis=1)

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump({"model": model, "features": FEATURES}, MODEL_OUT)
    print(f"Saved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
