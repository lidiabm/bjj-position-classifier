import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from .config import RESULTS_DIR, EXPERIMENT_NAME
from .utils import ensure_dir

def main (): 
    training_results_dir = os.path.join(RESULTS_DIR, EXPERIMENT_NAME, "training")
    history_csv_path = os.path.join(training_results_dir, "history.csv")
    history = pd.read_csv(history_csv_path)

    # Mejor epoch
    best_val_loss_epoch = int(history["val_loss"].idxmin())
    best_val_acc_epoch = int(history["val_accuracy"].idxmax())

    summary = {
        "best_epoch_by_val_loss": int(history.loc[best_val_loss_epoch, "epoch"]),
        "best_val_loss": float(history.loc[best_val_loss_epoch, "val_loss"]),
        "best_epoch_by_val_accuracy": int(history.loc[best_val_acc_epoch, "epoch"]),
        "best_val_accuracy": float(history.loc[best_val_acc_epoch, "val_accuracy"]),
        "last_epoch": int(history["epoch"].iloc[-1]),
    }

    with open(os.path.join(training_results_dir, "history_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 1) Loss
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.axvline(history.loc[best_val_loss_epoch, "epoch"], linestyle="--", label="best val_loss epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "loss_curve.png"), bbox_inches="tight")
    plt.close()

    # 2) Accuracy
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["accuracy"], label="train_accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], label="val_accuracy")
    plt.axvline(history.loc[best_val_acc_epoch, "epoch"], linestyle="--", label="best val_accuracy epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "accuracy_curve.png"), bbox_inches="tight")
    plt.close()

    # 3) Top-3 accuracy
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["top_3_accuracy"], label="train_top3")
    plt.plot(history["epoch"], history["val_top_3_accuracy"], label="val_top3")
    plt.xlabel("Epoch")
    plt.ylabel("Top-3 accuracy")
    plt.title("Training vs validation top-3 accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "top3_curve.png"), bbox_inches="tight")
    plt.close()

    # 4) Learning rate
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["learning_rate"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning rate schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "learning_rate_curve.png"), bbox_inches="tight")
    plt.close()

    # 5) Generalization gap
    history["acc_gap"] = history["accuracy"] - history["val_accuracy"]
    history["loss_gap"] = history["val_loss"] - history["loss"]

    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["acc_gap"], label="accuracy gap (train - val)")
    plt.plot(history["epoch"], history["loss_gap"], label="loss gap (val - train)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Gap")
    plt.title("Generalization gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(training_results_dir, "generalization_gap.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()