import os
import tensorflow as tf

from .config import (
    TRAIN_CSV, VAL_CSV, DF_CSV, 
    MODELS_DIR, RESULTS_DIR,
    IMG_SIZE, BACKBONE, EXPERIMENT_NAME, RUN_DIR,
    LR_HEAD, EPOCHS_HEAD,
    LR_FINE, EPOCHS_FINE,
    DO_FINE_TUNING, FINE_TUNE_LAST_N
)
from .data import load_trainval_splits, build_label_mapping, add_labels, make_dataset, load_df
from .model import build_model, compile_model
from .utils import ensure_dir, save_model_artifacts


def build_callbacks(model_dir, training_results_dir, history_name="history.csv"):
    """
    Crea i retorna la llista de callbacks que s'utilitzaran durant l'entrenament.

    Els callbacks permeten modificar o monitoritzar el procés d'entrenament automàticament
    en cada epoch. En aquest cas s'utilitzen per:
    - parar l'entrenament si el model deixa de millorar
    - reduir el learning rate si la validació s'estanca
    - guardar automàticament el millor model obtingut
    - registrar les mètriques d'entrenament en un arxiu CSV

    Args:
        model_dir (str): Directori on es guardaran els artefactes de l'entrenament.

    Returns:
        list: Llista de callbacks de Keras que es passaran a `model.fit()`.  
    """

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(training_results_dir, history_name),
            append=False
        )
    ]


def unfreeze_backbone_last_layers(model, last_n=30):
    """
    Descongela només les últimes capes del backbone per al fine-tuning.
    """
    backbone = model.get_layer("backbone")
    backbone.trainable = True

    for layer in backbone.layers[:-last_n]:
        layer.trainable = False

    return model


def main():
    # Crea carperes de sortida  
    model_dir = os.path.join(MODELS_DIR, RUN_DIR)
    training_results_dir = os.path.join(RESULTS_DIR, RUN_DIR, "training")
    ensure_dir(model_dir)
    ensure_dir(training_results_dir)

    # Carrega els splits d'entrenament i validació (csv)
    train_df, val_df = load_trainval_splits(TRAIN_CSV, VAL_CSV)
    df = load_df(DF_CSV)

    # Crea el mapping de classes en train i converteix etiquetes string a enters
    all_classes = sorted(df["position"].unique())
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    train_df = add_labels(train_df, class_to_idx)
    val_df = add_labels(val_df, class_to_idx)

    # Crea datasets de TensorFlow (training=True: shuffle activat) 
    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    # Model inicial: backbone congelat
    model = build_model(
        num_classes=len(all_classes),
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        backbone_trainable=False
    )
    model = compile_model(model, lr=LR_HEAD)

    save_model_artifacts(
        model=model,
        run_dir=model_dir,
        config={
            "experiment_name": EXPERIMENT_NAME,
            "backbone": BACKBONE,
            "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
            "num_classes": len(all_classes),
            "do_fine_tuning": DO_FINE_TUNING,
            "fine_tune_last_n": FINE_TUNE_LAST_N if DO_FINE_TUNING else 0,
            "learning_rate_head": LR_HEAD,
            "epochs_head": EPOCHS_HEAD,
            "learning_rate_fine": LR_FINE if DO_FINE_TUNING else None,
            "epochs_fine": EPOCHS_FINE if DO_FINE_TUNING else 0,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy", "top_3_accuracy"],
        }
    )

    # FASE 1: entrenament de la capçalera
    callbacks_head = build_callbacks(
        model_dir,
        training_results_dir,
        history_name="history_head.csv"
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks_head
    )

    # FASE 2: fine-tuning opcional
    if DO_FINE_TUNING:
        model = unfreeze_backbone_last_layers(model, last_n=FINE_TUNE_LAST_N)
        model = compile_model(model, lr=LR_FINE)

        callbacks_fine = build_callbacks(
            model_dir,
            training_results_dir,
            history_name="history_fine.csv"
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINE,
            callbacks=callbacks_fine
        )


if __name__ == "__main__":
    main()