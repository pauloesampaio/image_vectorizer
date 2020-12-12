import numpy as np
from tensorflow.keras.layers import Input, AveragePooling1D, Reshape, Flatten
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_vectorizer_model(reduce_dimensionality=False):
    """Loads pre-trained Xception neural network to generate feature vectors.
    If reduce_dimensionality, generates a 512-dim vector instead of 2048-dim.

    Args:
        reduce_dimensionality (bool, optional): If true, generates 512-dim vector. Defaults to False.

    Returns:
        tf.keras.model: Model to generate vectors
    """
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
    """Loads generator flowing through the image folder.
    If use_classes, infer classes from parent folder.

    Args:
        paths_dataframe (str): Path to the images folder
        use_classes (bool, optional): If true, infer classes using parent folder. Defaults to False.

    Returns:
        ImageDataGenerator: Image generator
    """
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
    """Function to generate vectors from the images on a given folder.
    If use classes, infer class using parent folder name.
    If reduce_dimensionality, generates a 512-dim vector (instead of a 2048-dim)

    Args:
        paths_dataframe (str): path to images folder
        use_classes (bool, optional): If true, infer class using parent folder name. Defaults to False.
        reduce_dimensionality (bool, optional): If true, generates a smaller, 512-dim vector. Defaults to False.

    Returns:
        np.array, list, list: Array with vectors, list with file pathes and, if use_classes, list of labels
    """
    image_generator = load_image_generator(paths_dataframe, use_classes=use_classes)
    model = load_vectorizer_model(reduce_dimensionality)
    vectors = model.predict(
        image_generator,
        batch_size=128,
        verbose=1,
        use_multiprocessing=True,
    )
    paths = image_generator.filepaths
    if use_classes:
        labels = image_generator.labels
    else:
        labels = None
    return vectors, paths, labels
