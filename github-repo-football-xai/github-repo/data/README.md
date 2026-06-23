# Dataset

This project uses the **European Soccer Database** from Kaggle.

## Download Instructions

1. Go to: https://www.kaggle.com/datasets/hugomathien/soccer
2. Sign in / create a free Kaggle account
3. Click **Download** (downloads a `.zip` file)
4. Extract the zip and place `database.sqlite` inside this `data/` folder

Your folder should look like this:
```
data/
├── README.md          (this file)
└── database.sqlite    (downloaded from Kaggle — not included in this repo due to file size)
```

## Dataset Summary

| | |
|---|---|
| Format | SQLite (`.sqlite`) |
| Coverage | 11 European leagues, 2008/09 – 2015/16 seasons |
| Matches | 25,000+ |
| Players | 10,000+ |
| Source of team attributes | EA Sports FIFA series (updated weekly) |

The notebook in `notebooks/football_prediction_shap.ipynb` automatically connects to `database.sqlite` and extracts only the columns needed for this analysis (match results + team tactical attributes).
