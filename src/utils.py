import os
import json
from tensorflow import keras

def ensure_dir(path: str):
    """
    Crea un directori si no existeix. Per assegurar que carpetes com 
    "models/" o "outputs/" existeixen abans de guardar fitxers.

    Args:
        path (str): Ruta del directori a crear.
    """
    
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    """
    Guarda un objecte Python en format JSON. Útil per guardar mapping 
    de classes (class_to_idx), metadades d'un experiment o paràmetres usats 

    Args:
        obj: Objecte serialitzable a JSON (dict, list, etc.).
        path (str): Ruta del fitxer JSON de sortida.
    """

    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_model_artifacts (model, run_dir, config): 
    """
    Guarda diferents artefactes del model per documentar i reproduir un experiment.
    Aquesta funció crea una carpeta per a l'experiment (si no existeix) i guarda una imatge
    amb l'arquitectura del model, un arxiu de text amb el seu summary i un fitxer JSON amb
    la configuració de l'experiment.

    Args:
        model (tf.keras.Model): Model de Keras ja construït.
        run_dir (str): Directori on es guardaran els artefactes de l'experiment.
        config (dict): Diccionari amb la configuració de l'experiment.
    """
    ensure_dir(run_dir)

    # Guarda imatge de l'arquitectura
    keras.utils.plot_model(model, to_file=os.path.join(run_dir, "model_architecture.png"), show_shapes=True, show_layer_names=True, expand_nested=True, dpi=120)

    # Guardar summary
    with open(os.path.join(run_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Guardar config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

