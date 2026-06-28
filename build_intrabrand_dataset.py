# construirea dataset-ului intra-brand

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Users\ruxiii\OneDrive\Desktop\disertatie\Bluetooth Datasets\Dataset 250 Msps")

chosen_folders = {
    "iphone4s": BASE_DIR / "Iphone" / "4s" / "013004004984503_oguz_guler",
    "iphone5": BASE_DIR / "Iphone" / "5" / "013409009258565_gamze_uyuk",
    "iphone5s": BASE_DIR / "Iphone" / "5s" / "359261061140526_melisa_topcu",
    "iphone6": BASE_DIR / "Iphone" / "6" / "354427066558690_mustafa_guclu",
    "iphone7": BASE_DIR / "Iphone" / "7" / "356563081643675_cuneyt_buyukkilic",
    "iphone7plus": BASE_DIR / "Iphone" / "7plus" / "355373083202269_akin_yavuz",
}

# transform semnalul brut in feature-uri statistice
def extract_features(x):
    x = np.asarray(x)
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "median": np.median(x),
        "q25": np.percentile(x, 25),
        "q75": np.percentile(x, 75),
        "rms": np.sqrt(np.mean(x**2)),
        "abs_mean": np.mean(np.abs(x)),
        "energy": np.sum(x**2),
    }

rows = []

# pentru fiecare label in folder, luam primele 20 de fisiere (pentru control) si calculam feature-urile
for label, folder in chosen_folders.items():
    files = sorted(folder.glob("*.txt"))
    print(f"{label}: {len(files)} fisiere gasite")

    for f in files[:20]:
        signal = np.loadtxt(f)
        feats = extract_features(signal)
        feats["label"] = label
        feats["file_name"] = f.name
        rows.append(feats)

df = pd.DataFrame(rows)

df.to_csv("intraband_dataset.csv", index=False)