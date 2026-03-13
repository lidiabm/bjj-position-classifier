
IMG_SIZE = (224, 224)       # tamany al que es redimensionen totes les imatges abans de ficar-les al model
BATCH_SIZE = 32             # número  d'imatges que el model procesa a la vegada en cada pas de l'entrenament
SEED = 42                   # llavor aleatòria 

SPLITS_DIR = "data/splits"  # directori on es troben els splits 
TRAIN_CSV = f"{SPLITS_DIR}/train.csv"
VAL_CSV   = f"{SPLITS_DIR}/val.csv"
TEST_CSV  = f"{SPLITS_DIR}/test.csv"

MODELS_DIR = "models"       # directori on es troben els models 
RESULTS_DIR = "results"     # directori on es troben els resultats del model en en conjunt de test 

EXPERIMENT_NAME = "efficientnetv2b0_baseline2_data_aug" # nom del experiment
BACKBONE = "EfficientNetV2B0"    # backbone del model (quina CNN preentrenada s'usarà com a base)

LR = 1e-3                   # learning rate
EPOCHS = 100                # quantes vegades el model veu el dataset
