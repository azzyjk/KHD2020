import tensorflow as tf # Tensorflow 2
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
def cnn_example():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5),
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    input_shape=(512,512, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(rate = 0.2))

    model.add(tf.keras.layers.Conv2D(16, (3, 3),
                                    kernel_initializer='he_normal',
                                    activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,
                                    kernel_initializer='he_normal',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'relu'))
    return model

def cnn():
    d_input = layers.Input(shape=(512, 512, 3))
    x = layers.ZeroPadding2D(padding=((5, 5), (5, 5)))(d_input)
    x = layers.Conv2D(filters=64, kernel_size=12, strides=3, use_bias=True, kernel_initializer=initializers.he_normal())(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1)(x)

    # 1차 분기_1
    div1_1 = layers.Conv2D(filters=128, kernel_size=8, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div1_1 = layers.BatchNormalization(axis=3)(div1_1)
    div1_1 = layers.Activation(activation='relu')(div1_1)
    div1_1 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div1_1)
    div1_1 = layers.BatchNormalization(axis=3)(div1_1)

    # 1차 분기_2
    div1_2 = layers.Conv2D(filters=64, kernel_size=6, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div1_2 = layers.BatchNormalization(axis=3)(div1_2)

    # 다시 합침
    x = layers.Add()([div1_1, div1_2, x])
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 2차 분기 (2개의 skipped connection을 가진 분기 3개)
    # (64 -> 128, 128) , (32 -> 64 -> 128, 32->128), (128 1개)

    # 2-1
    div2_1_1 = layers.Conv2D(filters=64, kernel_size=8, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div2_1_1 = layers.BatchNormalization(axis=3)(div2_1_1)
    div2_1_1 = layers.Activation(activation='relu')(div2_1_1)
    div2_1_1 = layers.Conv2D(filters=128, kernel_size=8, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div2_1_1)
    div2_1_1 = layers.BatchNormalization(axis=3)(div2_1_1)

    div2_1_2 = layers.Conv2D(filters=128, kernel_size=8, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div2_1_2 = layers.BatchNormalization(axis=3)(div2_1_2)

    div2_1 = layers.Add()([div2_1_1, div2_1_2])
    div2_1 = layers.Activation(activation='relu')(div2_1)

    # 2-2
    div2_2_1 = layers.Conv2D(filters=32, kernel_size=8, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div2_2_1 = layers.BatchNormalization(axis=3)(div2_2_1)
    div2_2_1 = layers.Activation(activation='relu')(div2_2_1)
    div2_2_1 = layers.Conv2D(filters=64, kernel_size=8, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div2_2_1)
    div2_2_1 = layers.BatchNormalization(axis=3)(div2_2_1)
    div2_2_1 = layers.Activation(activation='relu')(div2_2_1)
    div2_2_1 = layers.Conv2D(filters=128, kernel_size=8, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div2_2_1)
    div2_2_1 = layers.BatchNormalization(axis=3)(div2_2_1)

    div2_2_2 = layers.Conv2D(filters=32, kernel_size=8, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div2_2_2 = layers.BatchNormalization(axis=3)(div2_2_2)
    div2_2_2 = layers.Activation(activation='relu')(div2_2_2)
    div2_2_2 = layers.Conv2D(filters=128, kernel_size=8, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div2_2_2)
    div2_2_2 = layers.BatchNormalization(axis=3)(div2_2_2)

    div2_2 = layers.Add()([div2_2_1, div2_2_2])
    div2_2 = layers.Activation(activation='relu')(div2_2)

    # 2-3
    div2_3 = layers.Conv2D(filters=128, kernel_size=8, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div2_3 = layers.BatchNormalization(axis=3)(div2_3)
    div2_3 = layers.Activation(activation='relu')(div2_3)

    # x = layers.Add()([div2_1, div2_2, div2_3])

    # 3층
    # 2층에서 만든 3개의 레이어 중 2개만 선별하여 3개를 만든 뒤에 activation을 거침
    div3_1 = layers.Add()([div2_1, div2_2])
    div3_1 = layers.Dense(units=128, activation='relu')(div3_1)
    div3_1 = layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div3_1)
    div3_1 = layers.BatchNormalization(axis=3)(div3_1)

    div3_2 = layers.Add()([div2_1, div2_3])
    div3_2 = layers.Dense(units=128, activation='relu')(div3_2)
    div3_2 = layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div3_2)
    div3_2 = layers.BatchNormalization(axis=3)(div3_2)

    div3_3 = layers.Add()([div2_2, div2_3])
    div3_3 = layers.Dense(units=128, activation='relu')(div3_3)
    div3_3 = layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div3_3)
    div3_3 = layers.BatchNormalization(axis=3)(div3_3)

    x = layers.Add()([div3_1, div3_2, div3_3])
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 4층 - 가장 일반적인 resnet 구조 채용
    div4_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    div4_1 = layers.BatchNormalization(axis=3)(div4_1)
    div4_1 = layers.Activation(activation='relu')(div4_1)
    div4_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div4_1)
    div4_1 = layers.BatchNormalization(axis=3)(div4_1)
    div4_1 = layers.Activation(activation='relu')(div4_1)
    div4_1 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(div4_1)
    div4_1 = layers.BatchNormalization(axis=3)(div4_1)

    x = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x)

    x = layers.Add()([div4_1, x])
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 5층 - 추후 생각하기로 하고, 일단 이 구조를 밀고나가자.
    x = layers.Conv2D(filters=1024, kernel_size=2, strides=2, padding='SAME', use_bias=True, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=300, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(units=30, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(units=2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[d_input], outputs=x)
    model.summary()
    # tf.keras.utils.plot_model(model, 'model.png', True, True, 'TB', True, 200)

    return model

if __name__ == "__main__":
    cnn()