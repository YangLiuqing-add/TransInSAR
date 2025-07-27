from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from Uilts import *
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
from keras.models import load_model
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load noisy input data
noisy_data = load_and_reshape_dataset('noisy_train_data.h5', 'noisy_train_data', (20000, 9, 48, 48, 1))
noisy_train_data = noisy_data[:15000]
noisy_test_data = noisy_data[15000:19000]
noisy_tt = noisy_data[19000:]
noisy_tt = np.array(noisy_tt)

# Load DEM data
dem_data = load_and_reshape_dataset('dem_data.h5', 'dem_data', (20000, 1, 48, 48, 1))
dem_train = dem_data[:15000]
dem_test = dem_data[15000:19000]
dem_tt = dem_data[19000:]
dem_tt = np.array(dem_tt)

# Load ground truth displacement data
deformation_data = load_and_reshape_dataset('train_data.h5', 'train_data', (20000, 1, 48, 48, 1))
train_data = deformation_data[:15000]
test_data = deformation_data[15000:19000]
test_tt = deformation_data[19000:]
test_tt = np.array(test_tt)

# Load trained model
model = load_model(
    'Epochs/best_model.h5',
    custom_objects={
        'PatchEncoder': PatchEncoder,
        'VolumePatches': VolumePatches,
        'StochasticDepth': StochasticDepth,
        'combined_loss': spc_loss
    }
)

# Predict using test inputs
predictions = model.predict([noisy_tt, dem_tt])
predictions = np.squeeze(predictions)
test_tt = np.squeeze(test_tt)

print('test_tt.shape:', test_tt.shape)

# Compute average SSIM
ssim_scores = []
for i in range(test_tt.shape[0]):
    score = ssim(test_tt[i], predictions[i], data_range=test_tt[i].max() - test_tt[i].min())
    ssim_scores.append(score)

average_ssim = np.mean(ssim_scores)
print("Average SSIM (Transformer):", average_ssim)

