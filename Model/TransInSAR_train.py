
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from Uilts import *

# Load and reshape synthetic noisy timeseries dataset (dimension 9*48*48, 9 time epochs)
noisy_data = load_and_reshape_dataset('noisy_train_data.h5', 'noisy_train_data', (20000, 9, 48, 48, 1))

# Split synthetic noisy dataset into training, testing sets
noisy_train_data = noisy_data[:15000]
noisy_test_data = noisy_data[15000:19000]

# Load and reshape DEM dataset
dem_data = load_and_reshape_dataset('dem_data.h5', 'dem_data', (20000, 1, 48, 48, 1))

# Split DEM dataset into training, testing sets
dem_train = dem_data[:15000]
dem_test = dem_data[15000:19000]

# Load and reshape synthetic ground truth dataset
deformation_data = load_and_reshape_dataset('train_data.h5', 'train_data', (20000, 1, 48, 48, 1))

# Split deformation dataset into training, testing sets
train_data = deformation_data[:15000]
test_data = deformation_data[15000:19000]

### Hyperparameters

w1 = 48
w2 = 48

learning_rate = 0.001
weight_decay = 0.0001
drop_rate = 0.1
stochastic_depth_rate = 0.1
image_size = w1  # We'll resize input images to this size
projection_dim = 48
num_heads = 2
transformer_units = [projection_dim*2,projection_dim]
transformer_layers = 3

## Main model

input = layers.Input(shape=(9, 48, 48, 1))
input2 = layers.Input(shape=(1, 48, 48, 1))


def create_vit():
    x11 = convF1(input, 12, (2, 3, 3), 0.05)
    x11 = convF1(x11, 24, (2, 3, 3), 0.05)
    x11 = convF1(x11, 48, (2, 3, 3), 0.05)
    print('x11.shape', x11.shape)

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    for i in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(x11)
        print(x1.shape)

        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0)(x1, x1)
        print('attention_output.shape', attention_output.shape)
        attention_output = StochasticDepth(dpr[i])(attention_output)  #

        x2 = Add()([attention_output, x11])
        print('x2.shape', x2.shape)

        x3 = LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0)

        x3 = StochasticDepth(dpr[i])(x3)  #
        print('x3.shape', x3.shape)

        x11 = Add()([x3, x2])
    representation = LayerNormalization(epsilon=1e-6)(x11)
    print('representation.shape', representation.shape)
    inter = add([representation, input2])

    x = layers.Conv3DTranspose(48, (1, 3, 3), strides=1, padding="same")(inter)
    x = Activation(tf.nn.gelu)(x)
    x = layers.Conv3DTranspose(24, (1, 3, 3), strides=1, padding="same")(x)
    x = Activation(tf.nn.gelu)(x)
    x = layers.Conv3DTranspose(12, (1, 3, 3), strides=1, padding="same")(x)
    x = Activation(tf.nn.gelu)(x)

    output = Conv3D(filters=1, kernel_size=(1, 3, 3), activation='linear', padding='same')(x)

    model = Model(inputs=[input, input2], outputs=[output])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=spc_loss)
    model.summary()
    return model


def run_experiment(model):
    checkpoint = ModelCheckpoint(os.path.join('./Epochs/' + '{epoch:d}.h5'),
                                                       monitor='val_loss',
                                                       save_best_only=True,
                                                       verbose=1)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=15)

    callbacks = [checkpoint, reduce_lr, early_stopping_monitor]
    history = model.fit([noisy_train_data, dem_train], train_data, epochs=100,
                        validation_data=([noisy_test_data, dem_test], test_data), batch_size=32, callbacks=callbacks)
    model.save('./Epochs/' + 'last.h5')
    return history

import time

if __name__ == '__main__':
    start_time = time.time()
    create = create_vit()
    history = run_experiment(create)
    end_time = time.time()
    # 计算训练时间
    training_time = end_time - start_time
    print("Training time is：", training_time, "s")