from pathlib import Path

import cv2
import random
import numpy as np
from keras.src.callbacks import Callback
from keras.src.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenetv3

from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import register_keras_serializable


# ------------------------- CUSTOM DATA GENERATOR -------------------------
class AlbumentationsDataGenerator(Sequence):
    """
    Data generator using Albumentations. Allows loading from a folder or directly
    from a provided list of (path, label) samples.
    """

    def __init__(self, folder_path=None, samples_list=None,
                 batch_size=32, img_size=(224, 224),
                 shuffle=True, augment=None, min_per_class=0,
                 mean=None, std=None, oversample=False,
                 backbone="efficientnetb0", return_filenames=False):

        self.folder_path = Path(folder_path) if folder_path else None
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.oversample = oversample
        self.min_per_class = min_per_class
        self.mean = mean
        self.std = std
        self.backbone = backbone.lower()
        self.return_filenames = return_filenames

        # if sample list provided, then use it
        if samples_list is not None:
            self.samples = samples_list
            # obtain classes from labels
            labels = [label for _, label in self.samples]
            unique_labels = sorted(set(labels))
            self.class_indices = {cls: i for i, cls in enumerate(unique_labels)}
            self.classes = unique_labels

        else:
            # load from folder
            self.classes = sorted([d.name for d in self.folder_path.iterdir() if d.is_dir()])
            self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
            self.samples = []

            exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            class_samples = {
                cls: sum([list((self.folder_path / cls).glob(e)) for e in exts], [])
                for cls in self.classes
            }

            for cls in self.classes:
                imgs = class_samples.get(cls, [])
                if len(imgs) == 0:
                    print(f"[WARNING] Class '{cls}' has no images")
                    continue
                if self.oversample:
                    n_repeat = max(1, int(np.ceil(self.min_per_class / len(imgs))))
                    self.samples += [(p, self.class_indices[cls]) for p in imgs for _ in range(n_repeat)]
                else:
                    self.samples += [(p, self.class_indices[cls]) for p in imgs]

        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X, y, filenames = [], [], []

        for img_path, label in batch_samples:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.img_size)
                image = image.astype(np.float32) / 255.0

                if self.mean is not None and self.std is not None:
                    image = (image - self.mean) / self.std

                if self.augment:
                    augmented = self.augment(image=image)
                    image = augmented['image']

                X.append(image)
                y.append(label)
                filenames.append(str(img_path))
            except Exception as e:
                print(f"[ERROR] {img_path} -> {e}")

        if len(X) == 0:
            empty_X = np.zeros((0, *self.img_size, 3), dtype=np.float32)
            empty_y = np.zeros((0, len(self.classes)), dtype=np.float32)
            return (empty_X, empty_y, []) if self.return_filenames else (empty_X, empty_y)

        X = np.stack(X, axis=0).astype(np.float32)
        X = self._preprocess_backbone(X)
        y = to_categorical(y, num_classes=len(self.classes))

        if self.return_filenames:
            return X, y, filenames
        else:
            return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def get_labels(self):
        return [label for _, label in self.samples]

    def _preprocess_backbone(self, X):
        b = self.backbone
        X255 = X * 255.0

        if "resnet" in b:
            return preprocess_resnet(X255)
        elif "vgg" in b:
            return preprocess_vgg16(X255)
        elif "inception" in b:
            return preprocess_inception(X255)
        elif "efficientnet" in b:
            return preprocess_effnet(X255)
        elif "densenet" in b:
            return preprocess_densenet(X255)
        elif "mobilenetv2" in b:
            return preprocess_mobilenetv2(X255)
        elif "mobilenetv3" in b:
            return preprocess_mobilenetv3(X255)
        else:
            return X255

# ------------------------- CALLBACKS FOR DEALING WITH MINORITY CLASSES -------------------------

class ClassMetricsCallback(Callback):
    """
    Callback to compute metrics for each class on validation data
    """

    def __init__(self, val_gen, class_names):
        """
        val_gen: validation data generator created using previous AlbumentationsDataGenerator
        class_names: list of class names
        """
        super().__init__()
        self.val_gen = val_gen
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        """
        Computes metrics for each class on validation data at the end of each epoch.
        The metrics considered are: precision, recall and f1 but averaged as macro,
        in order to balance the impact of different classes.
        """
        y_true, y_pred = [], []

        for i in range(len(self.val_gen)):
            X_val, y_val = self.val_gen[i]
            if X_val.shape[0] == 0:
                continue
            preds = self.model.predict(X_val, verbose=0)
            y_true.extend(np.argmax(y_val, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        if len(y_true) > 0 and len(y_pred) > 0:
            precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
            recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

            print(f"\n[CLASS METRICS]")
            for i, cls in enumerate(self.class_names):
                print(f"{cls}: Precision={precisions[i]:.3f}, Recall={recalls[i]:.3f}, F1={f1s[i]:.3f}")

            logs['val_precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            logs['val_recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            logs['val_f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

            print(
                f" — val_precision_macro: {logs['val_precision_macro']:.4f}"
                f" — val_recall_macro: {logs['val_recall_macro']:.4f}"
                f" — val_f1_macro: {logs['val_f1_macro']:.4f}\n"
            )
        else:
            print("validation metrics not computed (empty batch)")


# ------------------------- CUSTOM LOSS AND METRICS FOR UNBALANCED DATASET -------------------------
def weighted_categorical_crossentropy(weights):
    """
    Cross-entropy loss weighted using class imbalance: minority class has greater weight.
    weights: array de pesos (1D) con un valor por clase.
    """
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        unweighted_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weight_per_sample = tf.reduce_sum(weights * y_true, axis=-1)
        return unweighted_loss * weight_per_sample

    return loss

def recall_macro(y_true, y_pred, NUM_CLASSES=6):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    recalls = []
    for i in range(NUM_CLASSES):
        y_true_i = K.cast(K.equal(y_true, i), 'float32')
        y_pred_i = K.cast(K.equal(y_pred, i), 'float32')
        tp = K.sum(y_true_i * y_pred_i)
        fn = K.sum(y_true_i * (1 - y_pred_i))
        recall = tp / (tp + fn + K.epsilon())
        recalls.append(recall)
    return K.mean(tf.stack(recalls))


def precision_macro(y_true, y_pred, NUM_CLASSES=6):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    precisions = []
    for i in range(NUM_CLASSES):
        y_true_i = K.cast(K.equal(y_true, i), 'float32')
        y_pred_i = K.cast(K.equal(y_pred, i), 'float32')
        tp = K.sum(y_true_i * y_pred_i)
        fp = K.sum((1 - y_true_i) * y_pred_i)
        precision = tp / (tp + fp + K.epsilon())
        precisions.append(precision)
    return K.mean(tf.stack(precisions))


def f1_macro(y_true, y_pred, NUM_CLASSES=6):
    p = precision_macro(y_true, y_pred, NUM_CLASSES)
    r = recall_macro(y_true, y_pred, NUM_CLASSES)
    return 2 * (p * r) / (p + r + K.epsilon())

# ------------------------- CBAM BLOCK IMPLEMENTATION -------------------------

# SpatialPooling layer: computes average and max pooling across channels and concatenates them
@register_keras_serializable(package="Custom")
class SpatialPooling(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialPooling, self).__init__(**kwargs)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return tf.concat([avg_pool, max_pool], axis=-1)

    def get_config(self):
        return super().get_config()


# CBAM BLOCK: Convolutional Block Attention Module, with channel and spatial attention mechanisms

@register_keras_serializable(package="Custom")
def cbam_block(input_tensor, reduction_ratio=16):
    channel = input_tensor.shape[-1]

    # Channel Attention
    shared_dense_one = layers.Dense(channel // reduction_ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True)
    shared_dense_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True)

    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_refined = layers.Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    concat = SpatialPooling()(channel_refined)
    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same',
                                      activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    refined = layers.Multiply()([channel_refined, spatial_attention])
    return refined

def build_backbone_cbam(model_name="EfficientNetB0",
                        input_shape=(224,224,3),
                        num_classes=6,
                        weights="imagenet",
                        reduction_ratio=16,
                        dropout_rate=0.5):

    Backbone = getattr(tf.keras.applications, model_name)

    inp = layers.Input(shape=input_shape)

    backbone = Backbone(include_top=False,
                        weights=weights,
                        input_tensor=inp)

    # CBAM before classifier
    feat = cbam_block(backbone.output, reduction_ratio=reduction_ratio)

    # final head
    feat = layers.GlobalAveragePooling2D()(feat)
    feat = layers.Flatten()(feat)
    feat = layers.Dense(256, activation='relu')(feat)
    feat = layers.Dropout(dropout_rate)(feat)
    out = layers.Dense(num_classes, activation='softmax')(feat)

    model = Model(inputs=inp, outputs=out,
                  name=f"{model_name}_CBAM")

    return model, backbone

# ------------------------- PREPROCESSING FOR EACH BACKBONE -------------------------
def preprocess_for_backbone(img, backbone_name):
    """
    Apply the appropriate preprocessing to the image according to the model architecture.
    """
    X255 = img * 255.0
    b = backbone_name.lower()

    if "resnet" in b:
        return preprocess_resnet(X255)
    elif "vgg" in b:
        return preprocess_vgg16(X255)
    elif "inception" in b:
        return preprocess_inception(X255)
    elif "efficientnet" in b:
        return preprocess_effnet(X255)
    elif "densenet" in b:
        return preprocess_densenet(X255)
    elif "mobilenetv2" in b:
        return preprocess_mobilenetv2(X255)
    elif "mobilenetv3" in b:
        return preprocess_mobilenetv3(X255)
    else:
        print("ERROR:")
        return None
    

# ------------------------- AUX FUNCTION -------------------------
def get_true_labels_from_keras_generator(gen):
    y_true = []
    for i in range(len(gen)):
        _, y_batch = gen[i]
        y_true.extend(np.argmax(y_batch, axis=1))
    return np.array(y_true)