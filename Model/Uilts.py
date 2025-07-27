from tensorflow.keras import layers
import h5py
import numpy as np
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D,Flatten,MaxPooling2D,ZeroPadding2D,Concatenate,Subtract,Lambda,Softmax,GlobalAveragePooling1D,MaxPooling1D,SpatialDropout1D,ReLU, Dense, Activation,Reshape,BatchNormalization, add, Embedding,Conv1D,LayerNormalization,MultiHeadAttention,Add,Dropout,Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import os

w1 = 48
w2 = 48

learning_rate = 0.001
weight_decay = 0.0001
drop_rate = 0.1
stochastic_depth_rate = 0.1
num_classes = 1
input_shape = (w1, w2, 3)
image_size = w1
projection_dim = 48
num_heads = 2
transformer_units = [projection_dim*2,projection_dim]
transformer_layers = 3

def load_and_reshape_dataset(file_path, dataset_name, new_shape):
    with h5py.File(file_path, 'r') as hf:
        data = np.array(hf.get(dataset_name))
    return np.reshape(data, new_shape)

def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        #print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1);


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);
    X = np.array(X)
    return X[:,:,0]

def yc_patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            # print(i1,i2)
            #       [i1,i2,ids]
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask;
    A = A[0:n1, 0:n2]
    return A

def yc_patch_with_gnss_flag_pos(A, l1, l2, o1, o2, x_gnss, y_gnss):
    n1, n2 = np.shape(A)
    tmp = np.mod(n1 - l1, o1)
    if tmp != 0:
        A = np.concatenate([A, np.zeros((o1 - tmp, n2))], axis=0)
    tmp = np.mod(n2 - l2, o2)
    if tmp != 0:
        A = np.concatenate([A, np.zeros((A.shape[0], o2 - tmp))], axis=-1)

    N1, N2 = np.shape(A)
    X = []
    flags = []
    gnss_positions = []  # 保存 GNSS 在 patch 内的坐标
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            patch = np.reshape(A[i1:i1+l1, i2:i2+l2], (l1*l2, 1))
            contains_gnss = (i1 <= y_gnss < i1+l1) and (i2 <= x_gnss < i2+l2)
            if contains_gnss:
                dy = y_gnss - i1
                dx = x_gnss - i2
            else:
                dy, dx = -1, -1
            X.append(patch)
            flags.append(contains_gnss)
            gnss_positions.append((dy, dx))
    X = np.array(X)
    flags = np.array(flags)
    gnss_positions = np.array(gnss_positions)
    return X[:,:,0], flags, gnss_positions


def convF1(inpt, d1, fil_ord, Dr):

    filters = int(inpt.shape[-1])

    pre = Conv3D(filters, fil_ord, strides=(1,1,1), padding='same', dilation_rate=1)(inpt)
    pre = Activation(tf.nn.gelu)(pre)

    pred = Conv3D(filters, fil_ord, strides=(1,1,1),padding='same', dilation_rate=2)(pre)
    pred = Activation(tf.nn.gelu)(pred)

    inf = Conv3D(filters, fil_ord,strides=(1,1,1), padding='same', dilation_rate=4)(pred)
    inf = Activation(tf.nn.gelu)(inf)
    inf = Add()([inf, inpt])

    inf1 = Conv3D(d1, fil_ord, strides=(1,1,1), padding='same', dilation_rate=1)(inf)
    inf1 = Activation(tf.nn.gelu)(inf1)
    encode = Dropout(Dr)(inf1)

    encode = layers.MaxPooling3D((3, 1, 1), padding="same")(encode)
    return encode

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None, dros_prop=None, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        # 自动兼容老模型的参数
        if drop_prob is not None:
            self.drop_prob = drop_prob
        elif dros_prop is not None:
            self.drop_prob = dros_prop
        else:
            raise ValueError("Must provide drop_prob (or legacy dros_prop).")

    def call(self, x, training=None):
        if training:
            keep_prob = 1.0 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            binary_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * binary_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'drop_prob': self.drop_prob
        })
        return config



class VolumePatches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):
        patches = tf.compat.v1.extract_volume_patches(
            inputs,
            ksizes=[1, 1, self.patch_size, self.patch_size, 1],
            strides=[1, 1, self.patch_size, self.patch_size, 1],
            padding='VALID'
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


image_size = 48
patch_size = 6
num_patches = (image_size // patch_size)


class PatchEncoder(Layer):

    def __init__(self, num_patches, projection_dim, **kwargs):
        # def __init__(self):
        super().__init__(**kwargs)

        self.num_patches = num_patches

        self.projection = Dense(units=projection_dim)

        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': num_patches,
            'projection_dim': projection_dim,
            # 'num_patches': num_patches,
        })
        return config

    def call(self, patch):
        # def call(self):
        # 定义call方法，用于前向传播
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # 生成一个位置矩阵
        encoded = self.projection(patch) + self.position_embedding(positions)
        # 对输入的patch进行特征投影，再加上嵌入的位置信息
        return encoded



def gradient_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, 48, 48, 1])
    y_pred = tf.reshape(y_pred, [-1, 48, 48, 1])

    true_grad = tf.image.sobel_edges(y_true)
    pred_grad = tf.image.sobel_edges(y_pred)

    loss = K.mean(K.abs(true_grad - pred_grad))
    return loss


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def physical_constraint_loss(y_true, y_pred):
    grad_loss = gradient_loss(y_true, y_pred)
    smooth_loss = K.mean(K.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))  # 二阶梯度
    return grad_loss + 0.1 * smooth_loss  # 0.1 权重可调


def spc_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    phys_loss = physical_constraint_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    return mse + 0.4 * phys_loss + ssim

def masked_combined_loss(y_true, y_pred, mask):
    mse = K.mean(mask * K.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)

    phys_loss = physical_constraint_loss(y_true, y_pred)

    return mse + 0.4 * phys_loss + ssim
    
def Loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    rmse = tf.sqrt(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))
    nrmse = tf.reduce_mean((rmse - tf.reduce_mean(rmse)) / tf.math.reduce_std(y_pred))
    print(1 - ssim, nrmse)
    return 1 - ssim + nrmse

def custom_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)

    std = tf.math.reduce_std(y_true)
    nrmse = tf.reduce_mean(rmse / (std + 1e-6))  # avoid division by zero

    return ssim_loss + nrmse

def masked_combined_loss(y_true, y_pred, mask):
    mse = K.mean(mask * K.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    phys_loss = physical_constraint_loss(y_true, y_pred)
    return mse + 0.4 * phys_loss + ssim

def mc_dropout_predict(model, inputs, T=5):
    preds = [model(inputs, training=True) for _ in range(T)]
    pred_stack = tf.stack(preds, axis=0)
    mean_pred = tf.reduce_mean(pred_stack, axis=0)
    std_pred = tf.math.reduce_std(pred_stack, axis=0)
    return mean_pred, std_pred

def generate_warmup_mask(std_pred, epoch, max_epoch, base_thresh=0.2, final_thresh=0.05):
    current_thresh = base_thresh - (base_thresh - final_thresh) * min(epoch / max_epoch, 1.0)
    std_norm = (std_pred - tf.reduce_min(std_pred)) / (tf.reduce_max(std_pred) - tf.reduce_min(std_pred) + 1e-8)
    return tf.cast(std_norm < current_thresh, tf.float32)

def gaussian_weight_map(shape, center, sigma):
    y, x = np.indices(shape)
    dy = y - center[0]
    dx = x - center[1]
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    

input = layers.Input(shape=(9,48,48,1))
input2 = layers.Input(shape=(1,48,48,1))

def create_vit():
    
    x11 = convF1(input, 12, (2,3,3), 0.05)
    x11 = convF1(x11, 24, (2,3,3), 0.05)
    x11 = convF1(x11, 48, (2,3,3), 0.05)
    print('x11.shape',x11.shape)

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    for i in range(transformer_layers):
    
        x1 = LayerNormalization(epsilon=1e-6)(x11)
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0)(x1, x1)
        print('attention_output.shape',attention_output.shape)
        attention_output = StochasticDepth(dpr[i])(attention_output) #

        x2 = Add()([attention_output, x11])
        x3 = LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0)

        x3 = StochasticDepth(dpr[i])(x3) #
        x11 = Add()([x3, x2])
    representation = LayerNormalization(epsilon=1e-6)(x11)
    print('representation.shape',representation.shape)

    inter = add([representation, input2])

    x = layers.Conv3DTranspose(48,(1,3,3), strides=1, padding="same")(inter)
    x = Activation(tf.nn.gelu)(x)
    x = layers.Conv3DTranspose(24,(1,3,3), strides=1, padding="same")(x)
    x = Activation(tf.nn.gelu)(x)
    x = layers.Conv3DTranspose(12,(1,3,3), strides=1, padding="same")(x)
    x = Activation(tf.nn.gelu)(x)

    output = Conv3D(filters=1, kernel_size=(1,3,3),  activation='linear', padding='same')(x)
    
    model = Model(inputs=[input,input2], outputs=[output])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,loss=spc_loss)
    model.summary()
    return model
