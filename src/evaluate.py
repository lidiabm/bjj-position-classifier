import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score)
from .config import TEST_CSV, MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR, EXPERIMENT_NAME
from .data import add_labels, make_dataset
from .utils import ensure_dir

# Paleta i estil global, perquè totes les figures tinguin una aparença consistent
PLOT_STYLE = "seaborn-v0_8-whitegrid"
CMAP_MAIN = "Blues"

plt.style.use(PLOT_STYLE)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "#C7D0D9",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.7,
    "grid.color": "#AEBAC6",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3):
    """
    Calcula la top-k accuracy: si la classe real està entre les k
    classes amb probabilitat més alta.

    Args:
        y_true (np.ndarray): Etiquetes reals (N,).
        y_prob (np.ndarray): Probabilitats softmax (N, C).
        k (int): Valor de k.

    Returns:
        float: Top-k accuracy.
    """
    k = min(k, y_prob.shape[1])

    # Obté les k classes amb més probabilitat i comproba si la classe real està entre elles
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    hits = np.any(top_k == y_true[:, None], axis=1)

    # Calcula el porcentatge d'encerts 
    return float(np.mean(hits))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Calcula l'Expected Calibration Error (ECE) a partir de la confiança
    màxima de softmax i la correcció top-1.

    Args:
        y_true (np.ndarray): Etiquetes reals (N,).
        y_prob (np.ndarray): Probabilitats softmax (N, C).
        n_bins (int): Nombre de bins.

    Returns:
        float: Valor de l'ECE.
    """

    # Obté la confiança i predicció i es comprova que aquesta és correcta
    conf = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    correct = (y_pred == y_true).astype(np.float64)

    # Crea intervals de confiança (bins)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]

        # Analitza cada bin (mostres on cada confiança cau dintre de l'interval)
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        # Calcula mètriques dintre del bin 
        if np.any(mask):
            bin_acc = np.mean(correct[mask])                    # precisió real en el bin 
            bin_conf = np.mean(conf[mask])                      # confiança mitjana del model 
            bin_weight = np.sum(mask) / n                       # proporció de mostres en aquets bin
            ece += np.abs(bin_acc - bin_conf) * bin_weight      # error ponderat

    return float(ece)


def save_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, n_bins: int = 10):
    """
    Aquesta funció dibuixa un reliability diagram (calibration curve) i mostra l'ECE.
    És a dir, crea una gràfica per veure si el model està ben calibrat i la guarda com a imatge (PNG).

    Args:
        y_true (np.ndarray): Etiquetes reals (N,).
        y_prob (np.ndarray): Probabilitats softmax (N, C).
        out_path (str): Ruta de sortida del PNG.
        n_bins (int): Nombre de bins.
    """
    ensure_dir(os.path.dirname(out_path))

    # Obté la confiança i predicció i es comprova que aquesta és correcta 
    conf = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    correct = (y_pred == y_true).astype(np.float64)

    # Divideix les prediccions en grups de confiança 
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_counts = []

    # Calcula la precisió en cada grup
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if np.any(mask):
            bin_accs.append(np.mean(correct[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0.0)
            bin_counts.append(0)

    # Calcula el ECE: quant s'allunya la confiança de la precisió real 
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    plt.style.use(PLOT_STYLE)
    cmap = plt.get_cmap(CMAP_MAIN)
    bar_color = cmap(0.60)
    line_color = "#D62728"

    plt.figure(figsize=(6.2, 6.2))
    width = (1.0 / n_bins) * 0.88

    plt.bar(bin_centers, bin_accs, width=width, alpha=0.92, color=bar_color, edgecolor="white", linewidth=0.8, label="Precisió per bin")
    plt.plot([0, 1], [0, 1], "--", color=line_color, linewidth=1.6, label="Calibració perfecta")
    plt.xlabel("Confiança")
    plt.ylabel("Precisió")
    plt.title(f"Reliability diagram (ECE = {ece:.4f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


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
    n_cols = max(1, len(df.columns))
    fig_h = min(0.42 * n_rows + 2.4, 20)
    fig_w = min(max(10, 1.25 * n_cols + 3), 18)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=16)

    table = ax.table(cellText=cell_text, colLabels=df.columns, rowLabels=df.index, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.42)

    # Estil visual
    cmap = plt.get_cmap(CMAP_MAIN)
    header_color = cmap(0.82)
    row_color_light = "#F7FAFD"
    row_color_alt = "#EAF2FB"
    edge_color = "#D7E3F0"
    index_color = "#DCEAF7"

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # capçalera
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.08)
        elif col == -1:
            cell.set_facecolor(index_color)
            cell.set_text_props(weight="semibold", color="#243746")
        elif row % 2 == 0:  # zebra stripes
            cell.set_facecolor(row_color_light)
        else:
            cell.set_facecolor(row_color_alt)

        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.6)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
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

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=CMAP_MAIN)
    ax.set_title("Matriu de confusió" + (" (normalitzada)" if normalize else ""))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.6)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # valors dins la matriu (opcional; en 18 classes encara és llegible)
    fmt = ".2f" if normalize else ".0f"
    thresh = cm_plot.max() * 0.62 if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(
                j, i, format(cm_plot[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "#1F2D3A",
                fontsize=7
            )
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Etiqueta prevista")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_metrics_per_class(report_df: pd.DataFrame, class_names: list, out_path: str, sort_by: str = "f1-score", highlight_n_worst: int = 5):
    """
    Guarda un gràfic de barre amb precision, recall y F1-score per classe,
    ordenant per mètrica y resaltant les pitjors classes.

    Args:
        report_df (pd.DataFrame): DataFrame del classification_report.
        class_names (list): Llista de classes reals.
        out_path (str): Ruta de sortida (PNG).
        sort_by (str): Mètrica utilitzada per ordenar les classes.
        highlight_n_worst (int): Número de pitjors classes a resaltar.
    """
    ensure_dir(os.path.dirname(out_path))

    # Es queda amb les classes reals 
    per_class = report_df.loc[class_names].copy()
    if "support" in per_class.columns:
        per_class["support"] = per_class["support"].astype(int)
    
    # Ordena les classes de pitjor a millor segons la mètrica escollida 
    per_class = per_class.sort_values(by=sort_by, ascending=True)

    # Extreu els valors a dibuixar 
    sorted_names = per_class.index.tolist()
    precision = per_class["precision"].values
    recall = per_class["recall"].values
    f1 = per_class["f1-score"].values
    support = per_class["support"].values if "support" in per_class.columns else None

    x = np.arange(len(sorted_names))
    width = 0.23

    cmap = plt.get_cmap(CMAP_MAIN)
    base_colors = [cmap(0.42), cmap(0.60), cmap(0.78)]
    highlight_color = "#C44E52"

    worst_classes = set(sorted_names[:highlight_n_worst])

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(16.5, 7.2))

    bars1 = ax.bar(x - width, precision, width, label="Precision", color=base_colors[0], edgecolor="white", linewidth=0.7)
    bars2 = ax.bar(x, recall, width, label="Recall", color=base_colors[1], edgecolor="white", linewidth=0.7)
    bars3 = ax.bar(x + width, f1, width, label="F1-score", color=base_colors[2], edgecolor="white", linewidth=0.7)

    if support is not None:
        xtick_labels = [f"{cls}\n(n={sup})" for cls, sup in zip(sorted_names, support)]
    else:
        xtick_labels = sorted_names

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Precision, Recall i F1-score per classe (ordenat per {sort_by})")

    # Es mantenen els colors base i s'anoten només les pitjors classes 
    for i, cls in enumerate(sorted_names):
        if cls in worst_classes:
            ax.scatter(x[i] + width, min(f1[i] + 0.035, 1.04), s=38, color=highlight_color, zorder=5)
            ax.text(x[i] + width, min(f1[i] + 0.065, 1.06), "pitjor", ha="center", va="bottom", fontsize=8, color=highlight_color, fontweight="semibold")
 
    legend_handles = [
        Patch(facecolor=base_colors[0], label="Precision"),
        Patch(facecolor=base_colors[1], label="Recall"),
        Patch(facecolor=base_colors[2], label="F1-score"),
        Patch(facecolor=highlight_color, label=f"{highlight_n_worst} pitjors classes")
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=4, loc="upper left")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
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

    # Calcula la confiança de cada precisió i comprova si encerta o falla
    conf = np.max(y_prob, axis=1)
    correct = (y_true == y_pred)

    cmap = plt.get_cmap(CMAP_MAIN)
    colors = [cmap(0.45), cmap(0.78)]

    bins = np.linspace(0, 1, 31)

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # Fa dos histogrames, un per quan el model encerta i un altre per quan falla 
    ax.hist(conf[correct], bins=bins, alpha=0.82, label="Correctes", color=colors[0], edgecolor="white", linewidth=0.7)
    ax.hist(conf[~correct], bins=bins, alpha=0.74, label="Incorrectes", color=colors[1], edgecolor="white", linewidth=0.7)
    
    ax.set_xlabel("Confiança (max softmax)")
    ax.set_ylabel("Freqüència")
    ax.set_title("Distribució de confiança: correctes vs incorrectes")
    ax.legend(frameon=False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    # Directoris de sortida
    run_name = EXPERIMENT_NAME    # canvia segons l'experiment 
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
    labels = list(range(num_classes))

    # Carregar test i crear dataset
    test_df = pd.read_csv(TEST_CSV)
    test_df = add_labels(test_df, class_to_idx)
    test_ds = make_dataset(test_df, training=False)

    # Carregar model entrenat 
    model_path = os.path.join(MODELS_DIR, EXPERIMENT_NAME, "best_model.keras")
    model = tf.keras.models.load_model(model_path)

    # Evaluació bàsica amb Keras (loss + accuracy)
    loss, keras_acc, keras_top3 = model.evaluate(test_ds, verbose=0)

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
    top3_acc = top_k_accuracy(y_true, y_prob, k=3)
    top5_acc = top_k_accuracy(y_true, y_prob, k=5)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    # Report per classe
    report_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Matriu de confusió
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Guardar outputs 
        # Mètriques globals en un JSON
    metrics = {
        "model_path": model_path,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "keras_accuracy": float(keras_acc),
        "keras_top3": float(keras_top3),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "ece": float(ece),
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
    preds_out["true_class_confidence"] = y_prob[np.arange(len(y_true)), y_true]
    preds_out["correct"] = (preds_out["label"].values == preds_out["pred_label"].values)

    # Guardem també el top-3 de classes previstes per facilitar l'anàlisi d'errors
    top3_idx = np.argsort(y_prob, axis=1)[:, -3:][:, ::-1]
    preds_out["top3_pred_labels"] = [list(row) for row in top3_idx]
    preds_out["top3_pred_positions"] = [[idx_to_class[i] for i in row] for row in top3_idx]

    preds_out.to_csv(os.path.join(preds_dir, "test_predictions.csv"), index=False)

        # Matriu de confusió normal i normalitzada
    save_confusion_matrix(cm, class_names, os.path.join(figures_dir, "confusion_matrix.png"), normalize=False)
    save_confusion_matrix(cm, class_names, os.path.join(figures_dir, "confusion_matrix_norm.png"), normalize=True)

        # F1 per classe (gràfic de barres)
    save_metrics_per_class(
        report_df,
        class_names,
        os.path.join(figures_dir, "f1_precision_recall_per_class.png"),
        sort_by="f1-score",
        highlight_n_worst=5
    )
        
        # Histograma de confiança: correctes vs incorrectes
    save_confidence_histograms(y_prob, y_true, y_pred, os.path.join(figures_dir, "confidence_hist.png"))

        # Reliability diagram / calibration curve
    save_reliability_diagram(
        y_true,
        y_prob,
        os.path.join(figures_dir, "reliability_diagram.png"),
        n_bins=10
    )
    
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
    print(f"- {os.path.join(figures_dir, 'reliability_diagram.png')}")


if __name__ == "__main__":
    main()