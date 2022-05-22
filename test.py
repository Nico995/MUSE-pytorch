from glob import glob

import numpy as np
import pandas as pd

import muse_pytorch as muse

# Morpho
morph_features = np.load('../MUSE_sample33/ST_morph_features.npy')
orig_names = sorted(glob('../MUSE_sample33/ST_crops_33/*.png'))

orig_barcodes = []

for name in list(orig_names):
    crop = name.split('/')[-1]
    sample = crop.split('_')[0].replace('ST', '')
    barcode = crop.split('_')[-1].split('.')[0]

    orig_barcodes.append(f'sample{sample}_{barcode}-1_1')

morph_data = pd.DataFrame(columns=orig_barcodes, data=morph_features.T)
morph_labels = pd.read_csv(
    '../MUSE_sample33/SM-morpho-clusters_PCA-100_res-0,4.csv')['Morph']

df_trans_features = pd.read_csv('../MUSE_sample33/ST33_trans_features_500.csv')
df_trans_features = df_trans_features.set_index('Unnamed: 0')
trans_features = df_trans_features.transpose().values

trans_labels = pd.read_csv(
    '../MUSE_sample33/SM-trans-clusters_500_PCA-100_res-0,4.csv')['Trans']

z, x_hat, y_hat, latent_x, latent_y = muse.fit_predict(trans_features,
                                                       morph_features,
                                                       trans_labels,
                                                       morph_labels,
                                                       init_epochs=3,
                                                       refine_epochs=3,
                                                       cluster_epochs=5,
                                                       cluster_update_epoch=2,
                                                       joint_latent_dim=50)

print(z.shape)
print(x_hat.shape)
print(y_hat.shape)
print(latent_x.shape)
print(latent_y.shape)
# Unpacking prediction outputs
