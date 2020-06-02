import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
import csv
from model import AlexNet


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


def load_label(label):
    indices = [0, 1, 2, 3, 4, 5]  #  number of output kinds
    depth = len(indices)
    return tf.one_hot(indices, depth)[label]  #  one-hot encoding


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
        loss_display = tf.reduce_sum(loss)
        tf.print("loss : ", loss_display)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test_step(images, labels):
    predictions = model(images, training=False)
    predictions = np.argmax(predictions.numpy(), axis=1)
    labels = np.argmax(labels.numpy(), axis=1)
    accuracy = sum([1 if x==y else 0 for x, y in zip(predictions, labels)]) / predictions.shape[0]

    return accuracy

#  create dataset
with open("../newgame.csv") as f:
    reader = csv.reader(f)
    labels_list = [row for row in reader]

labels_dic = {}

for l_el in np.array(labels_list[1:]):
    labels_dic[l_el[0]] = l_el[1]

img_path = "../data/"
files = os.listdir(img_path)
files_file = [f for f in files if os.path.isfile(os.path.join(img_path, f))]  #  all image names
random.shuffle(files_file)
files_file_custom = [img_path+files_file[i] for i in range(len(files_file))]  #  all image path
all_labels = [int(labels_dic[name]) for name in files_file]

SPLIT_NUM = 0.7
split_idx = int(0.7*len(files_file))

train_image_path_ds = tf.data.Dataset.from_tensor_slices(files_file_custom[0:split_idx])
train_image_ds = train_image_path_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_labels_ds = tf.data.Dataset.from_tensor_slices(all_labels[0:split_idx])
train_labels_ds = train_labels_ds.map(load_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_image_path_ds = tf.data.Dataset.from_tensor_slices(files_file_custom[split_idx:])
test_image_ds = test_image_path_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_labels_ds = tf.data.Dataset.from_tensor_slices(all_labels[split_idx:])
test_labels_ds = test_labels_ds.map(load_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_labels_ds))
test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_labels_ds))


#  training
model = AlexNet()
optimizer = tf.keras.optimizers.Adam()

EPOCHS = 1
BATCH_SIZE = 32
IMAGE_SIZE = len(files_file)

for epoch in range(EPOCHS):
    print("{} / {} epoch ...".format(epoch, EPOCHS))
    checkpoint_dir = "./path"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if epoch % 10 == 0:
        checkpoint_prefix = os.path.join(checkpoint_dir, "{}.ckpt".format(epoch))
        root = tf.train.Checkpoint(optimizer=optimizer)
        root.save(checkpoint_prefix)

    for images, labels in train_image_label_ds.batch(BATCH_SIZE).shuffle(BATCH_SIZE):
        train_step(images, labels)

    accuracy_list = []
    for images, labels in test_image_label_ds.batch(BATCH_SIZE).shuffle(BATCH_SIZE):
        accuracy_tmp = test_step(images, labels)
        accuracy_list.append(accuracy_tmp)

    accuracy = sum(accuracy_list) / len(accuracy_list)
    tf.print("accuracy : ", accuracy)
