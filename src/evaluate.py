import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score)
from .config import TEST_CSV, MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR
from .data import add_labels, make_dataset
from .utils import ensure_dir

def save_table_as_image(df: pd.DataFrame, path: str, title: str | None = None):
    """
    Guarda un DataFrame com a imatge (PNG) amb una taula llegible.

    Args:
        df (pd.DataFrame): Taula a convertir a imatge.
        path (str): Ruta de sortida (PNG).
        title (str | None): Títol opcional de la figura.
    """
    ensure_dir(os.path.dirname(path))

    # Convertim tots els valors a string amb format uniforme
    cell_text = []
    for row in df.values:
        formatted_row = []
        for v in row:
            if isinstance(v, (int, float, np.number)):
                formatted_row.append(f"{v:.3f}")
            else:
                formatted_row.append(str(v))
        cell_text.append(formatted_row)

    # Ajustar mida de la figura segons el nombre de files
    n_rows = max(1, len(df))
    fig_h = min(0.45 * n_rows + 2.5, 20)

    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    table = ax.table(
        cellText=cell_text,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Estil visual
    for (row, col), cell in table.get_celld().items():

        if row == 0:            
            cell.set_facecolor("#40466e")   # capçalera
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:      
            cell.set_facecolor("#f2f2f2")   # zebra stripes
        cell.set_edgecolor("#dddddd")       # bordes suaus

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str, normalize: bool = False):
    """
    Guarda una matriu de confusió com a imatge (PNG).

    Args:
        cm (np.ndarray): Matriu de confusió
        class_names (list): Noms de les classes (ordre = índex de label).
        out_path (str): Ruta on guardar el PNG.
        normalize (bool): Si True, normalitza per fila (recall per classe).
    """
    ensure_dir(os.path.dirname(out_path))

    # Convertim a float per poder normalitzar sense divisió entera 
    cm_plot = cm.astype(np.float64)

    # Si hi ha normalització, es normalitza per files: cada fila suma 1 (si hi ha mostres de la classe)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums != 0)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # valors dins la matriu (opcional; en 18 classes encara és llegible)
    fmt = ".2f" if normalize else ".0f"
    thresh = cm_plot.max() * 0.6 if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            plt.text(
                j, i, format(cm_plot[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=7
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_metrics_per_class(report_df: pd.DataFrame, class_names: list, out_path: str):
    """
    Guarda un gràfic de barres amb precision, recall i F1-score per classe.

    Args:
        report_df (pd.DataFrame): DataFrame del classification_report.
        class_names (list): Llista de classes reals.
        out_path (str): Ruta de sortida (PNG).
    """
    ensure_dir(os.path.dirname(out_path))

    per_class = report_df.loc[class_names]

    precision = per_class["precision"].values
    recall = per_class["recall"].values
    f1 = per_class["f1-score"].values

    x = np.arange(len(class_names))
    width = 0.25

    cmap = plt.get_cmap("GnBu")
    colors = [cmap(0.35), cmap(0.55), cmap(0.75)]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(14,6))
    plt.bar(x - width, precision, width, label="Precision", color=colors[0])
    plt.bar(x, recall, width, label="Recall", color=colors[1])
    plt.bar(x + width, f1, width, label="F1-score", color=colors[2])

    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Precision, Recall i F1-score per classe")
    plt.ylim(0,1)

    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_confidence_histograms(y_prob: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """
    Guarda un histograma de la confiança (max softmax) separant encerts vs errors.

    Args:
        y_prob (np.ndarray): Probabilitats softmax (N, C).
        y_true (np.ndarray): Etiquetes reals (N,).
        y_pred (np.ndarray): Etiquetes predites (N,).
        out_path (str): Ruta de sortida (PNG).
    """
    ensure_dir(os.path.dirname(out_path))

    conf = np.max(y_prob, axis=1)
    correct = (y_true == y_pred)

    cmap = plt.get_cmap("GnBu")
    colors = [cmap(0.35), cmap(0.75)]
    
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7, 4))
    plt.hist(conf[correct], bins=30, alpha=0.7, label="Correctes", color=colors[0])
    plt.hist(conf[~correct], bins=30, alpha=0.7, label="Incorrectes", color=colors[1])
    plt.xlabel("Confiança (max softmax)")
    plt.ylabel("Freqüència")
    plt.title("Distribució de confiança: correctes vs incorrectes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # Directoris de sortida
    run_name = "baseline_efficientnetb0_224"    # canvia segons l'experiment 
    run_dir = os.path.join(RESULTS_DIR, run_name)

    figures_dir = os.path.join(run_dir, "figures")
    metrics_dir = os.path.join(run_dir, "metrics")
    preds_dir   = os.path.join(run_dir, "predictions")
    ensure_dir(figures_dir)
    ensure_dir(metrics_dir)
    ensure_dir(preds_dir)

    # Carregar mapping de classes, assegurant que la codificació és consistent amb l'entrenament
    mapping_path = os.path.join(OUTPUTS_DIR, "class_to_idx.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    
    # s'inverteix el mapping (int: posició), per obtenir una llista ordenada de noms de posicions segons l’índex.
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}  
    num_classes = len(idx_to_class)                             
    class_names = [idx_to_class[i] for i in range(num_classes)] 

    # Carregar test i crear dataset
    test_df = pd.read_csv(TEST_CSV)
    test_df = add_labels(test_df, class_to_idx)
    test_ds = make_dataset(test_df, training=False)

    # Carregar model entrenat 
    model_path = os.path.join(MODELS_DIR, "baseline_efficientnetb0.keras")
    model = tf.keras.models.load_model(model_path)

    # Evaluació bàsica amb Keras (loss + accuracy)
    loss, keras_acc = model.evaluate(test_ds, verbose=0)

    # Prediccions 
    y_prob = model.predict(test_ds, verbose=0)  # retorna la probabilitat per classe per cada imatge
    y_pred = np.argmax(y_prob, axis=1)          # escull l'índex de la classe amb major probabilitat             
    y_true = test_df["label"].values            # agafa les etiquetes reals de les imatges                 

    # Mètriques globals 
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Report per classe
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Matriu de confusió
    cm = confusion_matrix(y_true, y_pred)

    # Guardar outputs 
        # Mètriques globals en un JSON
    metrics = {
        "model_path": model_path,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "keras_accuracy": float(keras_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "num_test_samples": int(len(test_df)),
        "num_classes": int(num_classes),
    }
    with open(os.path.join(metrics_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(metrics_dir,"metrics.csv"), index=False)
    metrics_df_num = metrics_df.drop(columns=["model_path", "run_name"], errors="ignore")
    save_table_as_image(metrics_df_num, os.path.join(metrics_dir, "metrics.png"), title="Mètriques globals del model (test)")

    
        # Report per classe (CSV + PNG)
    report_df.to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    save_table_as_image(report_df, os.path.join(metrics_dir, "classification_report.png"), title="Classification report (per classe)")

        # Prediccions per mostra (per anàlisi d’errors)
    preds_out = test_df[["image_path", "position", "label"]].copy()
    preds_out["pred_label"] = y_pred
    preds_out["pred_position"] = [idx_to_class[i] for i in y_pred]
    preds_out["pred_confidence"] = np.max(y_prob, axis=1)
    preds_out["correct"] = (preds_out["label"].values == preds_out["pred_label"].values)
    preds_out.to_csv(os.path.join(preds_dir, "test_predictions.csv"), index=False)

        # Matriu de confusió normal i normalitzada
    save_confusion_matrix(cm, class_names, os.path.join(figures_dir, "confusion_matrix.png"), normalize=False)
    save_confusion_matrix(cm, class_names, os.path.join(figures_dir, "confusion_matrix_norm.png"), normalize=True)

        # F1 per classe (gràfic de barres)
    save_metrics_per_class(report_df, class_names, os.path.join(figures_dir, "f1_precision_recall_per_class.png"))
        
        # Histograma de confiança: correctes vs incorrectes
    save_confidence_histograms(y_prob, y_true, y_pred, os.path.join(figures_dir, "confidence_hist.png"))
    
    # Imprimir resum per consola 
    print("\n Resultats en test guardats a: ")
    print(f"- {os.path.join(metrics_dir, 'test_metrics.json')}")
    print(f"- {os.path.join(metrics_dir, 'metrics.csv')}")
    print(f"- {os.path.join(metrics_dir, 'metrics.png')}")
    print(f"- {os.path.join(metrics_dir, 'classification_report.csv')}")
    print(f"- {os.path.join(metrics_dir, 'classification_report.png')}")
    print(f"- {os.path.join(preds_dir, 'test_predictions.csv')}")
    print(f"- {os.path.join(figures_dir, 'confusion_matrix.png')}")
    print(f"- {os.path.join(figures_dir, 'confusion_matrix_norm.png')}")
    print(f"- {os.path.join(figures_dir, 'f1_precision_recall_per_class.png')}")
    print(f"- {os.path.join(figures_dir, 'confidence_hist.png')}")


if __name__ == "__main__":
    main()