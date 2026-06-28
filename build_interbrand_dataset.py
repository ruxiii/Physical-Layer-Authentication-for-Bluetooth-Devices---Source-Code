# construirea dataset-ului inter-brand

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Users\ruxiii\OneDrive\Desktop\disertatie\Bluetooth Datasets\Dataset 250 Msps")

chosen_folders = {
    "xiaomi_mi6": BASE_DIR / "Xiaomi" / "Mi6" / "864890030385966_umut_erkan",
    "iphone6": BASE_DIR / "Iphone" / "6" / "354427066558690_mustafa_guclu",
    "samsung_s5": BASE_DIR / "Samsung" / "S5" / "353812060996205_melisa_oktem",
    "lg_g4": BASE_DIR / "LG" / "G4" / "352334073607175_mert_kilic",
    "sony_xperiam5": BASE_DIR / "Sony" / "XperiaM5" / "354188076543367_leman_cetindere",
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

df.to_csv("interbrand_dataset.csv", index=False)