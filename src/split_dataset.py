import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "data"
IMG_PATH = os.path.join(DATASET_PATH, "raw")
ANNOT_PATH = os.path.join(DATASET_PATH, "annotations.json")
OUTPUT_SPLITS = "data/splits"

os.makedirs(OUTPUT_SPLITS, exist_ok=True)

# Carregar JSON
with open(ANNOT_PATH, "r") as f:
    annotations = json.load(f)

# Convertir a DataFrame
samples = []
for ann in annotations:
    img_file = os.path.join(IMG_PATH, ann["image"] + ".jpg")
    samples.append({
        "image_path": img_file,
        "pose1": ann.get("pose1", None),
        "pose2": ann.get("pose2", None),
        "position": ann["position"],
        "frame": ann["frame"]
    })

df = pd.DataFrame(samples)
print("Mostres totals:", len(df))


# Split en train, validation i test  
train_val, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["position"]
)
train, val = train_test_split(
    train_val,
    test_size=0.20,  
    random_state=42,
    stratify=train_val["position"]
)

# Guardar CSVs
df.to_csv(os.path.join(OUTPUT_SPLITS, "df.csv"), index=False)
train.to_csv(os.path.join(OUTPUT_SPLITS, "train.csv"), index=False)
val.to_csv(os.path.join(OUTPUT_SPLITS, "val.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_SPLITS, "test.csv"), index=False)

print("Splits generats:")
print(f"  Train: {len(train)}")
print(f"  Validation:   {len(val)}")
print(f"  Test:  {len(test)}")