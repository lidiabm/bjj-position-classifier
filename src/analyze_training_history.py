import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from .config import RESULTS_DIR, EXPERIMENT_NAME, RUN_DIR
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

def load_history(training_results_dir: str) -> pd.DataFrame:
    """
    Carrega l'historial d'entrenament.

    Casos suportats:
    - Entrenament normal: history.csv
    - Entrenament en dues fases: history_head.csv i history_fine.csv

    Retorna un únic DataFrame amb una columna extra 'phase'.
    """
    history_path = os.path.join(training_results_dir, "history.csv")
    history_head_path = os.path.join(training_results_dir, "history_head.csv")
    history_fine_path = os.path.join(training_results_dir, "history_fine.csv")

    # En cas d'entrenament estàndard: 
    if os.path.exists(history_path):
        history = pd.read_csv(history_path).copy()
        history["phase"] = "single"
        return history

    # En cas de fine-tuning en dues fases
    elif os.path.exists(history_head_path) and os.path.exists(history_fine_path):
        history_head = pd.read_csv(history_head_path).copy()
        history_fine = pd.read_csv(history_fine_path).copy()

        history_head["phase"] = "head"
        history_fine["phase"] = "fine"

        # Reajustem epochs perquè la gràfica sigui contínua
        if "epoch" in history_head.columns and "epoch" in history_fine.columns:
            last_head_epoch = int(history_head["epoch"].iloc[-1])
            history_fine["epoch"] = history_fine["epoch"] + last_head_epoch + 1

        history = pd.concat([history_head, history_fine], ignore_index=True)
        return history

    else:
        raise FileNotFoundError("No s'ha trobat cap historial compatible.")

def main():
    # Directori on es guarden els resultats relacionats amb l'entrenament
    training_results_dir = os.path.join(RESULTS_DIR, RUN_DIR, "training")
    ensure_dir(training_results_dir)

    history = load_history(training_results_dir)

    # Identifica la millor epoch segons val_loss i val_accuracy
    best_val_loss_epoch = int(history["val_loss"].idxmin())
    best_val_acc_epoch = int(history["val_accuracy"].idxmax())

    # Guarda un petit resum amb les epochs més rellevants
    summary = {
        "millor_epoch_segons_val_loss": int(history.loc[best_val_loss_epoch, "epoch"]),
        "millor_val_loss": float(history.loc[best_val_loss_epoch, "val_loss"]),
        "millor_epoch_segons_val_accuracy": int(history.loc[best_val_acc_epoch, "epoch"]),
        "millor_val_accuracy": float(history.loc[best_val_acc_epoch, "val_accuracy"]),
        "ultima_epoch": int(history["epoch"].iloc[-1]),
    }

    with open(os.path.join(training_results_dir, "history_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 1) Corba de pèrdua: entrenament vs validació
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["loss"], label="Pèrdua d'entrenament")
    plt.plot(history["epoch"], history["val_loss"], label="Pèrdua de validació")
    plt.axvline(
        history.loc[best_val_loss_epoch, "epoch"],
        linestyle="--",
        label="Millor epoch segons val_loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Pèrdua")
    plt.title("Pèrdua d'entrenament i validació")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "loss_curve.png"), bbox_inches="tight")
    plt.close()

    # 2) Corba d'accuracy: entrenament vs validació
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["accuracy"], label="Accuracy d'entrenament")
    plt.plot(history["epoch"], history["val_accuracy"], label="Accuracy de validació")
    plt.axvline(
        history.loc[best_val_acc_epoch, "epoch"],
        linestyle="--",
        label="Millor epoch segons val_accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy d'entrenament i validació")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "accuracy_curve.png"), bbox_inches="tight")
    plt.close()

    # 3) Corba de top-3 accuracy: entrenament vs validació
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["top_3_accuracy"], label="Top-3 accuracy d'entrenament")
    plt.plot(history["epoch"], history["val_top_3_accuracy"], label="Top-3 accuracy de validació")
    plt.xlabel("Epoch")
    plt.ylabel("Top-3 accuracy")
    plt.title("Top-3 accuracy d'entrenament i validació")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "top3_curve.png"), bbox_inches="tight")
    plt.close()

    # 4) Evolució del learning rate al llarg de les epochs
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["learning_rate"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Evolució del learning rate")
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "learning_rate_curve.png"), bbox_inches="tight")
    plt.close()

    # 5) Bretxa de generalització: diferència entre train i validació
    history["acc_gap"] = history["accuracy"] - history["val_accuracy"]
    history["loss_gap"] = history["val_loss"] - history["loss"]

    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["acc_gap"], label="Bretxa d'accuracy (train - val)")
    plt.plot(history["epoch"], history["loss_gap"], label="Bretxa de pèrdua (val - train)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Bretxa")
    plt.title("Bretxa de generalització")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "generalization_gap.png"), bbox_inches="tight")
    plt.close()

    print("\nAnàlisi de l'historial d'entrenament guardat a:")
    print(f"- {os.path.join(training_results_dir, 'history_summary.json')}")
    print(f"- {os.path.join(training_results_dir, 'loss_curve.png')}")
    print(f"- {os.path.join(training_results_dir, 'accuracy_curve.png')}")
    print(f"- {os.path.join(training_results_dir, 'top3_curve.png')}")
    print(f"- {os.path.join(training_results_dir, 'learning_rate_curve.png')}")
    print(f"- {os.path.join(training_results_dir, 'generalization_gap.png')}")


if __name__ == "__main__":
    main()