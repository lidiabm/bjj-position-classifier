import pandas as pd
import tensorflow as tf
from .config import IMG_SIZE, BATCH_SIZE, SEED

def load_trainval_splits(train_csv: str, val_csv: str):
    """
    Carrega els CSV dels splits del dataset (train, validation i test)
    Aquests CSV contenen, com a mínim: image_path (ruta de la imatge) i position (etiqueta de la posició)

    Args:
        train_csv (str): Ruta al CSV del conjunt d'entrenament.
        val_csv (str): Ruta al CSV del conjunt de validació.
        test_csv (str): Ruta al CSV del conjunt de test.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - train_df: DataFrame amb les mostres d'entrenament
            - val_df: DataFrame amb les mostres de validació
            - test_df: DataFrame amb les mostres de test
    """

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    return train_df, val_df


def build_label_mapping(train_df: pd.DataFrame):
    """
    Construeix el mapping de classes (string → enter).
    Com les xarxes neuronals no poden treballar amb strings com "mount1", 
    "guard2"..., cal convertir aquestes etiquetes en valors numèrics.
    Per tant, aquesta funció obté totes les classes, les ordena i crea 
    un diccionari.  
    
    Args:
        train_df (pd.DataFrame): DataFrame d'entrenament amb la columna "position".

    Returns:
        tuple[list[str], dict[str,int]]:
            - classes: llista ordenada de totes les classes
            - class_to_idx: diccionari que assigna un enter a cada classe
    """

    classes = sorted(train_df["position"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    return classes, class_to_idx


def add_labels(df: pd.DataFrame, class_to_idx: dict):
    """
    Afegeix una nova columna numèrica al DataFrame anomenada "label", que correspon 
    a la versió codificada de la columna "position" usant el mapping creat anteriorment.

    Args:
        df (pd.DataFrame): DataFrame amb la columna "position".
        class_to_idx (dict): Mapping de classes

    Returns:
        pd.DataFrame: Còpia del DataFrame amb la nova columna "label".
    """

    df = df.copy()
    df["label"] = df["position"].map(class_to_idx)

    # Comprovació de seguretat: si alguna etiqueta no existeix al mapping
    if df["label"].isna().any():
        missing = df[df["label"].isna()]["position"].unique().tolist()
        raise ValueError(f"Hi ha classes que no existeixen al mapping d'entrenament: {missing}")

    df["label"] = df["label"].astype(int)

    return df


def decode_resize(path: tf.Tensor, label: tf.Tensor):
    """
    Llegeix una imatge i aplica el preprocessat necessari, seguint els següents passos: 
        1. Llegeix el fitxer d'imatge
        2. Decodifica el JPEG
        3. Redimensiona la imatge
        4. Aplica el preprocessat específic

    Args:
        path (tf.Tensor): Ruta de la imatge.
        label (tf.Tensor): Etiqueta numèrica associada a la imatge.

    Returns:
        tuple[tf.Tensor, tf.Tensor]:
            - img: imatge preprocessada
            - label: etiqueta corresponent
    """

    # Llegir el fitxer d'imatge
    img = tf.io.read_file(path)

    # Convertir el JPEG en tensor d'imatge
    img = tf.image.decode_jpeg(img, channels=3)

    # Redimensionar la imatge al tamany definit
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    
    # Normalització específica del model EfficientNet
    # img = preprocess_input(img)

    # Convertim a float32
    img = tf.cast(img, tf.float32)      # per EfficientNetV2 

    return img, label


def make_dataset(df: pd.DataFrame, training: bool = False):
    """
    Converteix el DataFrame en un tf.data.Dataset, dataset que serà 
    usat per entrenar o avaluar el model.
    El pipeline realitza: càrrega d'imatges, preprocessat, batching i 
    prefetch per millorar el rendiment. I si training=True també s'aplica shuffle.

    Args:
        df (pd.DataFrame): DataFrame amb les columnes "image_path" i "label".
        training (bool): Indica si el dataset és per entrenament.

    Returns:
        tf.data.Dataset: Dataset preparat per ser usat durant l'entrenament del model.
    """

    # Crear dataset a partir de les rutes d'imatge i les etiquetes
    ds = tf.data.Dataset.from_tensor_slices(
        (df["image_path"].values, df["label"].values)
    )

    # Aplica la funció de càrrega i preprocessat
    ds = ds.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

    # Barreja dades si és el conjunt d'entrenament
    if training: ds = ds.shuffle(buffer_size=1000, seed=SEED, reshuffle_each_iteration=True)

    # Agrupa mostres en batches
    ds = ds.batch(BATCH_SIZE)

    # Prepara el següent batch mentre el model està entrenant
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds