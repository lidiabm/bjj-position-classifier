import os
import json
import pandas as pd

DATASET_PATH = "data"
IMG_PATH = os.path.join(DATASET_PATH, "raw")
ANNOT_PATH = os.path.join(DATASET_PATH, "annotations.json")
OUTPUT_SPLITS = "data/splits"

def get_combat_id(sequence_id: str) -> str:
    """
    Funció que retorna l'identificador del combat al qual pertany una seqüència.

    En aquest dataset, diverses seqüències de vídeo corresponen al mateix sparring,
    ja que un mateix combat ha estat gravat des de càmeres diferents.

    Per tant, aquesta funció agrupa manualment les seqüències que s'ha comprovat
    que pertanyen al mateix combat, per tal de poder fer la partició del dataset
    a nivell de combat i evitar fuga d'informació entre train, validation i test.

    Args:
        sequence_id (str): Identificador de la seqüència (2 primers dígits del nom de la imatge).

    Returns:
        str: Identificador del combat corresponent ("C1", "C2", ..., "C6").

    Raises:
        ValueError: Si la seqüència no correspon a cap combat conegut.
    """
    if sequence_id in ["00", "01", "02"]:
        return "C1"
    elif sequence_id in ["03", "04", "05"]:
        return "C2"
    elif sequence_id in ["06", "07", "08"]:
        return "C3"
    elif sequence_id in ["09", "10"]:
        return "C4"
    elif sequence_id in ["11", "12", "13"]:
        return "C5"
    elif sequence_id in ["14", "15"]:
        return "C6"
    else:
        raise ValueError(f"Secuencia desconocida: {sequence_id}")


def main(): 

    os.makedirs(OUTPUT_SPLITS, exist_ok=True)

    # Cargar JSON
    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # 
    # Convertir a DataFrame
    samples = []
    for ann in annotations:
        image_name = str(ann["image"])   
        sequence_id = image_name[:2]                # 2 primers dígits del nom de la imatge
        combat_id = get_combat_id(sequence_id)
        img_file = os.path.join(IMG_PATH, image_name + ".jpg")

        samples.append({
            "image": image_name,
            "image_path": img_file,
            "sequence_id": sequence_id,
            "combat_id": combat_id, 
            "pose1": ann.get("pose1", None),
            "pose2": ann.get("pose2", None),
            "position": ann["position"],
            "frame": ann["frame"]
        })

    df = pd.DataFrame(samples)

    print("Mostres totals:", len(df))
    print("Seqüències trobades:", sorted(df["sequence_id"].unique()))
    print("Combats trobats:", sorted(df["combat_id"].unique()))

    # Guardar dataframe complet
    df.to_csv(os.path.join(OUTPUT_SPLITS, "df.csv"), index=False)

    # Separar en per combats
    combat_dfs = {}
    for combat in sorted(df["combat_id"].unique()):
        combat_dfs[combat] = df[df["combat_id"] == combat].copy()
        print(f"{combat}: {len(combat_dfs[combat])} imágenes")

    # Definir els 3 experiments
    experiments = {
        "split1": {
            "train": ["C1", "C2", "C3", "C4"],
            "val":   ["C5"],
            "test":  ["C6"],
        },
        "split2": {
            "train": ["C1", "C2", "C5", "C6"],
            "val":   ["C3"],
            "test":  ["C4"],
        },
        "split3": {
            "train": ["C3", "C4", "C5", "C6"],
            "val":   ["C1"],
            "test":  ["C2"],
        }
    }

    # Crear carpetas y guardar CSVs
    for split_name, split_def in experiments.items():
        split_dir = os.path.join(OUTPUT_SPLITS, split_name)
        os.makedirs(split_dir, exist_ok=True)

        train = pd.concat([combat_dfs[c] for c in split_def["train"]], ignore_index=True)
        val   = pd.concat([combat_dfs[c] for c in split_def["val"]], ignore_index=True)
        test  = pd.concat([combat_dfs[c] for c in split_def["test"]], ignore_index=True)

        train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(split_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(split_dir, "test.csv"), index=False)

        print(f"\n{split_name}")
        print("  Train:", split_def["train"], "->", len(train), "imágenes")
        print("  Val:  ", split_def["val"], "->", len(val), "imágenes")
        print("  Test: ", split_def["test"], "->", len(test), "imágenes")

        print("  Clases en train:")
        print(train["position"].value_counts().sort_index())

if __name__ == "__main__":
    main()

# import os
# import json
# import pandas as pd
# from sklearn.model_selection import train_test_split

# DATASET_PATH = "data"
# IMG_PATH = os.path.join(DATASET_PATH, "raw")
# ANNOT_PATH = os.path.join(DATASET_PATH, "annotations.json")
# OUTPUT_SPLITS = "data/splits"

# os.makedirs(OUTPUT_SPLITS, exist_ok=True)

# # Carregar JSON
# with open(ANNOT_PATH, "r") as f:
#     annotations = json.load(f)

# # Convertir a DataFrame
# samples = []
# for ann in annotations:
#     img_file = os.path.join(IMG_PATH, ann["image"] + ".jpg")
#     samples.append({
#         "image_path": img_file,
#         "pose1": ann.get("pose1", None),
#         "pose2": ann.get("pose2", None),
#         "position": ann["position"],
#         "frame": ann["frame"]
#     })

# df = pd.DataFrame(samples)
# print("Mostres totals:", len(df))


# # Split en train, validation i test  
# train_val, test = train_test_split(
#     df,
#     test_size=0.20,
#     random_state=42,
#     stratify=df["position"]
# )
# train, val = train_test_split(
#     train_val,
#     test_size=0.20,  
#     random_state=42,
#     stratify=train_val["position"]
# )

# # Guardar CSVs
# df.to_csv(os.path.join(OUTPUT_SPLITS, "df.csv"), index=False)
# train.to_csv(os.path.join(OUTPUT_SPLITS, "train.csv"), index=False)
# val.to_csv(os.path.join(OUTPUT_SPLITS, "val.csv"), index=False)
# test.to_csv(os.path.join(OUTPUT_SPLITS, "test.csv"), index=False)

# print("Splits generats:")
# print(f"  Train: {len(train)}")
# print(f"  Validation:   {len(val)}")
# print(f"  Test:  {len(test)}")
