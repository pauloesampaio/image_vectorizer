import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def _load_vectorizer_model():
    i = Input([224, 224, 3], dtype=np.float32)
    x = preprocess_input(i)
    core = Xception(include_top=False, pooling="max")
    x = core(x)
    model = Model(inputs=[i], outputs=[x])
    return model


def _load_image_generator(file_path):
    image_dataset = ImageDataGenerator()
    image_generator = image_dataset.flow_from_directory(
        file_path,
        target_size=(224, 224),
        class_mode=None,
        batch_size=32,
        shuffle=False,
    )
    file_list = image_generator.filenames
    return {"file_list": file_list, "image_generator": image_generator}


def generate_vectors(file_path):
    file_list, image_generator = _load_image_generator(file_path).values()
    model = _load_vectorizer_model()
    vectors = model.predict(
        image_generator, batch_size=128, verbose=1, use_multiprocessing=True
    )
    return file_list, vectors
