import numpy as np
import pandas as pd

"""
Kodas skirtas išskirčių šalinimui iš A24_cleaned.csv naudojant IQR metodą.

    - ignoruoja penkis požymius (Steel_Plate_Thickness, Pixels_Areas, X_Perimeter, Y_Perimeter, Outside_X_Index);
    - logaritmuoja Sum_of_Luminosity (ką turbūt turėjom padaryt pirmam darbe)
    - apskaičiuoja IQR ribas (Q1 - 1.5·IQR, Q3 + 1.5·IQR) likusiems požymiams;
    - pažymi eilutes, kuriose bent vienas požymis išeina už ribų;
    - pašalina tokias eilutes ir išsaugo rezultatą į A24_no_outliers.csv;
    - atspausdina, kiek eilučių pašalinta ir pagal kuriuos požymius jos dažniausiai buvo pažymėtos.

"""

"""
Kodėl ignoruojam šituos:
    - Dubliuoja log-transformuotus požymius - LogOfAreas, Log_X_Index, Log_Y_Index jau yra šių dydžių logaritminės versijos. 
    - Steel_Plate_Thickness yra diskretus, netinka IQR, jų iš viso yra 25 skirtingos reikšmės

"""
IGNORED_FEATURES = [
    "Steel_Plate_Thickness",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Outside_X_Index",
]

INPUT_PATH = "A24_cleaned.csv"
OUTPUT_PATH = "A24_no_outliers.csv"

IQR_MULTIPLIER = 1.5
MIN_FLAGGED_FEATURES = 1


def detection_features(df):
    features = df.drop(columns="class").copy()
    features["Sum_of_Luminosity"] = np.log10(features["Sum_of_Luminosity"])
    return features.drop(columns=IGNORED_FEATURES)


def outlier_flags(df, k=IQR_MULTIPLIER):
    features = detection_features(df)
    q1 = features.quantile(0.25)
    q3 = features.quantile(0.75)
    iqr = q3 - q1
    below_range = features.lt(q1 - k * iqr)
    above_range = features.gt(q3 + k * iqr)
    return below_range | above_range


def rows_with_outliers(flags, min_feats=MIN_FLAGGED_FEATURES):
    return flags.sum(axis=1) >= min_feats


def drop_summary(flags, rows_to_drop):
    counts = flags[rows_to_drop].sum().astype(int)
    share = (counts / rows_to_drop.sum() * 100).round(1)
    summary = pd.DataFrame({
        "Požymis": counts.index,
        "Pašalinta eilučių": counts.to_numpy(),
        "Dalis nuo pašalintų, %": share.to_numpy(),
    })
    return summary.sort_values("Pašalinta eilučių", ascending=False, ignore_index=True)


df = pd.read_csv(INPUT_PATH)
flags = outlier_flags(df)
rows_to_drop = rows_with_outliers(flags)

kept = df[~rows_to_drop]
kept.to_csv(OUTPUT_PATH, index=False)

print(f"{len(df)} -> {len(kept)} (pašalinta {rows_to_drop.sum()}, {rows_to_drop.mean() * 100:.1f} %)")
print(f"\nPašalintos eilutės pagal požymį (k={IQR_MULTIPLIER}, min. požymių eilutėje={MIN_FLAGGED_FEATURES}):")
print(drop_summary(flags, rows_to_drop).to_string(index=False))