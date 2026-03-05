import tensorflow as tf
from .config import TRAIN_CSV, VAL_CSV, TEST_CSV, MODELS_DIR, OUTPUTS_DIR, LR, EPOCHS, IMG_SIZE
from .data import load_trainval_splits, build_label_mapping, add_labels, make_dataset
from .model import build_model, compile_model
from .utils import ensure_dir, save_json

def main():

    # Crea les carpetes de sortida (assegura que existeixen)
    ensure_dir(MODELS_DIR)
    ensure_dir(OUTPUTS_DIR)

    # Carrega els splits (csv). El test_df es carrega, però NO s'usa en l'entrenament, sinó en evaluate.py)
    train_df, val_df = load_trainval_splits(TRAIN_CSV, VAL_CSV)

    # Crea el mapping de classes en train i converteix etiquetes string a enters
    classes, class_to_idx = build_label_mapping(train_df)
    train_df = add_labels(train_df, class_to_idx)
    val_df = add_labels(val_df, class_to_idx)

    # Crea datasets de TensorFlow (training=True: shuffle activat) 
    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    # Construeix i compila el model
    model = build_model(num_classes=len(classes), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), backbone_trainable=False)
    model = compile_model(model, lr=LR)

    # Guarda mapping per avaluació i interpretació
    save_json(class_to_idx, f"{OUTPUTS_DIR}/class_to_idx.json")

    # Callbacks
    # - EarlyStopping: para quan no millora la validació
    # - ModelCheckpoint: guarda el millor model segons val_accuracy
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODELS_DIR}/baseline_efficientnetb0.keras",
            monitor="val_accuracy",
            save_best_only=True
        ),
    ]

    # Entrena
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)


if __name__ == "__main__":
    main()