import tensorflow as tf


def scheduler(epoch, lr):
    return lr * 0.95


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
