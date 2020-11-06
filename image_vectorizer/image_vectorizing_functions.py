import numpy as np
from tensorflow.keras.layers import Input, AveragePooling1D, Reshape, Flatten
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_vectorizer_model(reduce_dimensionality=False):
    i = Input([224, 224, 3], dtype=np.float32)
    x = preprocess_input(i)
    core = Xception(include_top=False, pooling="max")
    x = core(x)
    if reduce_dimensionality:
        x = Reshape((-1, 1))(x)
        x = AveragePooling1D(pool_size=(4,), strides=(4,))(x)
        x = Flatten()(x)
    model = Model(inputs=[i], outputs=[x])
    return model


def load_image_generator(paths_dataframe, use_classes=False):
    y_col, class_mode = None, None
    if use_classes:
        y_col = "class"
        class_mode = "raw"
    image_dataset = ImageDataGenerator()
    image_generator = image_dataset.flow_from_dataframe(
        dataframe=paths_dataframe,
        x_col="filename",
        y_col=y_col,
        target_size=(224, 224),
        class_mode=class_mode,
        batch_size=32,
        shuffle=False,
    )
    return image_generator


def generate_vectors(paths_dataframe, use_classes=False, reduce_dimensionality=False):
    image_generator = load_image_generator(paths_dataframe, use_classes=use_classes)
    model = load_vectorizer_model(reduce_dimensionality)
    vectors = model.predict(
        image_generator,
        batch_size=128,
        verbose=1,
        use_multiprocessing=True,
    )
    return vectors
