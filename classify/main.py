import os
from datetime import datetime

# pytorch GPU 版本自带CUDNN,可以便于tensorflow 使用 GPU
try:
    import torch
except Exception as _:
    pass

import keras
import tensorflow as tf
from keras import layers
from keras import models
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.mobilenet_v3 import MobileNetV3Small

# 检测GPU 是否可用
print(tf.test.is_gpu_available())

# 基础配置
data_dir_input = "datas"
batch_size = 64
train_val_split = 0.1
seed = 123

image_size = (224, 224)
# 图像增强:尽可能多的图像预处理,用于改善小数据过拟合
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomContrast(0.5),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
])

# 准备数据集
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir_input,
    seed=seed,
    image_size=image_size,
    validation_split=train_val_split,
    subset="training"
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir_input,
    seed=seed,
    image_size=image_size,
    validation_split=train_val_split,
    subset="validation"
)
class_names = train_dataset.class_names

# 定义网络部分
input_tensor = keras.Input(shape=(*image_size, 3))
output_tensor = layers.Rescaling(1. / 255)(input_tensor)  # 0-1
mobile_net_v3_small = MobileNetV3Small(include_top=False, input_tensor=output_tensor, include_preprocessing=False)
# mobile net v3 small
output_tensor = mobile_net_v3_small.output

# 分类
output_tensor = layers.core.Flatten()(output_tensor)
output_tensor = layers.Dropout(0.2)(output_tensor)
output_tensor = layers.Dense(128, activation="relu")(output_tensor)
output_tensor = layers.Dense(len(class_names), activation="softmax")(output_tensor)
classify_model = models.Model(input_tensor, output_tensor)  # 最终,重组成为模型

# 包括图像增强的全部网络
total_model = tf.keras.Sequential([
    data_augmentation,
    classify_model,
])

# 定义损失函数相关
total_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy"
    ]
)

# 定义tensorboard 日志目录
date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", f"{date_time}")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=True,
    write_steps_per_second=True
)

# 定义最优保存
check_save_path = os.path.join("best")
ckpt = tf.train.get_checkpoint_state("best")
if ckpt and ckpt.model_checkpoint_path:
    total_model.load_weights(ckpt.model_checkpoint_path)
    print("成功载入已训练模型")
else:
    print(f"未载入已训练模型,将从头开始训练")

cp_callback = ModelCheckpoint(
    check_save_path,
    save_best_only=True,
)

# 训练
total_model.fit(
    train_dataset,
    epochs=2,
    validation_data=val_dataset,
    validation_freq=1,
    callbacks=[
        tensorboard_callback,
        cp_callback
    ]
)
