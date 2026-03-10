import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_model(num_classes: int, input_shape=(224, 224, 3), backbone_trainable: bool = False):
    """
    Construeix el model de classificació basat en transfer learning amb MobileNetV2.
    En aquesta cas, usem MobileNetV2 preentrenada amb ImageNet com a extractor de característiques.
    A més, afegim una "capa de capçalera" (classification head) a sobre per classificar les 18 posicions.

    Args:
        num_classes (int): Nombre de classes a predir (18).
        input_shape (tuple): Forma de la imatge d'entrada (H, W, C).
        backbone_trainable (bool): Si False, es congelen els pesos d'MobileNetV2 (baseline).
                                   Si True, es permet el fine-tuning (entrenar també part del backbone).

    Returns:
        tf.keras.Model: Model llest per compilar i entrenar.
    """

    # Es carrega backbone preentrenat (MobileNetV2)
    base = MobileNetV2(
        weights="imagenet", 
        include_top=False, 
        input_shape=input_shape
    )

    # Es congelar o no el backbone (baseline vs fine-tuning)
    base.trainable = backbone_trainable

    # Es defineix l'entrada del model
    inputs = layers.Input(shape=input_shape)

    # Preprocessat específic de MobileNetV2
    x = preprocess_input(inputs)

    # Es passar la imatge pel backbone.
    x = base(x, training=False)

    # Es redueix el mapa de característiques a un vector (embedding)
    x = layers.GlobalAveragePooling2D()(x)

    # Es regularitza per reduir overfitting
    x = layers.Dropout(0.3)(x)

    # Capa final de classificació (softmax -> probabilitat per cada classe)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Es crea el model final
    model = Model(inputs, outputs)

    return model


def compile_model(model: tf.keras.Model, lr: float = 1e-3):
    """
    Compila el model amb l'optimitzador, la funció de pèrdua i les mètriques que es mostraran durant 
    l'entrenament. 

    Args:
        model (tf.keras.Model): Model creat amb build_model().
        lr (float): Learning rate per Adam. Per transfer learning, 1e-3 sol funcionar bé al baseline.

    Returns:
        tf.keras.Model: El mateix model, ja compilat.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),   # optimitzador és l'algoritme que actualitza els pesos  
        loss="sparse_categorical_crossentropy",                 # loss mesura quant de malament prediu el model 
        metrics=["accuracy"]                                    # metricas que s'usen per monitorizar el rendiment 
    )

    return model