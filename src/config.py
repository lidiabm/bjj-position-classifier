import os

IMG_SIZE = (224, 224)       # tamany al que es redimensionen totes les imatges abans de ficar-les al model
BATCH_SIZE = 32             # número  d'imatges que el model procesa a la vegada en cada pas de l'entrenament
SEED = 42                   # llavor aleatòria 

SPLIT_NAME = "split1"
SPLITS_DIR = os.path.join("data", "splits", SPLIT_NAME) # directori on es troben els splits 
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")
DF_CSV = os.path.join("data", "splits", "df.csv")

MODELS_DIR = "models"       # directori on es troben els models 
RESULTS_DIR = "results"     # directori on es troben els resultats del model en en conjunt de test 

EXPERIMENT_NAME = "efficientnetv2b0_finetuning"     # nom del experiment
RUN_DIR = os.path.join(EXPERIMENT_NAME, SPLIT_NAME) # directori de la execució
BACKBONE = "EfficientNetV2B0" # backbone del model (quina CNN preentrenada s'usarà com a base)

DO_FINE_TUNING = True
FINE_TUNE_LAST_N = 30

EPOCHS_HEAD = 15
EPOCHS_FINE = 30 if DO_FINE_TUNING else 0

LR_HEAD = 1e-3
LR_FINE = 1e-5 if DO_FINE_TUNING else 0