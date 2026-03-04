import os
import json

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