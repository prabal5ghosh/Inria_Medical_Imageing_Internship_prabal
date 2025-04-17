# %% [markdown]
# # Prabal Ghosh
# 
# MRI

# %%
import nibabel as nib
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# %% [markdown]
# ## Data loading 

# %%

# File paths
mri_path = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/mri.nii.gz")    # MRI image

mask_path = Path("/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/segmentations_total_mr/iliopsoas_right.nii.gz")   # Segmentation mask

# Load the images using nibabel
mri_img = nib.load(mri_path)
print(f"Loaded MRI image shape: {mri_img.shape}")
mask_img = nib.load(mask_path)
print(f"Loaded mask image shape: {mask_img.shape}")

# Get the image data as numpy arrays
mri_data = mri_img.get_fdata()
# print(f"MRI data : {mri_data}")
mask_data = mask_img.get_fdata()
# print(f"Mask data : {mask_data}")


# %%
mask_data[mask_data > 0].shape

# %% [markdown]
# ## test 3

# %% [markdown]
# ### find the segmented part from mri 

# %%
# Ensure mask is binary (0/1)
mask_data = (mask_data > 0).astype(np.float32)

# Extract region by multiplication
segmented_data = mri_data * mask_data  # Background = 0


# %%

# Save result
segmented_img = nib.Nifti1Image(segmented_data, mri_img.affine, mri_img.header)
nib.save(segmented_img, '/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/segmented_region_iliopsoas_right.nii.gz')

# %%
mask_img.header

# %%
mri_img.header

# %%
mri_img

# %%
mri_img.dataobj

# %%
segmented_img.header

# %%
print(mri_img.affine)
print()
print()

print(mask_img.affine)

# %%
segmented_img.affine

# %%
# (axial slice 6)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(mri_data[:, 6, :], cmap='gray')  # Original MRI
plt.title('Original MRI')
plt.subplot(1, 3, 2)
plt.imshow(mask_data[:, 6, :], cmap='Reds')  # Mask
plt.title('Mask')
plt.subplot(1, 3, 3)
plt.imshow(segmented_data[:, 6, :], cmap='gray')  # Extracted region
plt.title('Extracted Region')


plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/mask_gray_image.png', dpi=300)  # Save the plot
plt.show()


# %%
# (axial slice 6)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(mri_data[:, 6, :], )  # Original MRI
plt.title('Original MRI')
plt.subplot(1, 3, 2)
plt.imshow(mask_data[:, 6, :], )  # Mask
plt.title('Mask')
plt.subplot(1, 3, 3)
plt.imshow(segmented_data[:, 6, :], )  # Extracted region
plt.title('Extracted Region')
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/mask_image.png', dpi=300)  # Save the plot

plt.show()

# %% [markdown]
# ### 1. Intensity calculation 1.  Here coordinates are considered. ------------its useless.........don't use this

# %%

# Flatten the array and remove zeros (background)
intensities = segmented_data[segmented_data > 0]

# Verify
print(f"Number of voxels in segmented region: {len(intensities)}")
print(f"Min intensity: {np.min(intensities)}, Max intensity: {np.max(intensities)}")

# %%


# %%
import pandas as pd

# Create a DataFrame and save to CSV
df3 = pd.DataFrame(intensities, columns=['Intensity'])
df3.to_csv('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/segmented_intensities.csv', index=False)


# %%
df3.head(10)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df3['Intensity'], bins=100, color='skyblue', alpha=0.7, edgecolor='black',label='Intensity Distribution')
plt.title('Intensity Distribution of Segmented Region')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
# plt.grid(axis='y', alpha=0.3)


threshold = df3['Intensity'].quantile(0.90)
plt.axvline(threshold, color='red', linestyle='--',linewidth=2, label=f'90th percentile: {threshold:.2f}')
plt.axvline(np.mean(intensities), color='yellow', linestyle='dashed', label=f'Mean: {np.mean(intensities):.2f}')
plt.axvline(np.median(intensities), color='green', linestyle='dashed', label=f'Median: {np.median(intensities):.2f}')
plt.legend()

plt.tight_layout()
# plt.savefig('intensity_histogram.png', dpi=300)  # Save the plot

plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/intensity_histogram.png', dpi=300)  # Save the plot

plt.show()

# Print key stats
print(f"90th Percentile Threshold: {threshold:.2f}")
print(f"Mean Intensity: {np.mean(intensities):.2f}")
print(f"Median Intensity: {np.median(intensities):.2f} ")
print(f"Standard Deviation: {np.std(intensities):.2f} ")
print(f"min Intensity: {np.min(intensities):.2f} ")
print(f"max Intensity: {np.max(intensities):.2f} ")



# %% [markdown]
# ### 2. Intensity calculation 2. Here coordinates are considered.

# %%
# Load the segmented MRI data
# segmented_img = nib.load('segmented_region_3.nii.gz')
# segmented_data = segmented_img.get_fdata()



# binary mask  (True = segmented region) (where intensities > 0)
mask = segmented_data > 0

# Extract coordinates (x, y, z) and intensities of non-zero voxels
# Get voxel coordinates of the segmented region  -----  it extracts the 3D positions (x, y, z) of all voxels in the segmented region of the MRI.
coordinates = np.argwhere(mask)  # Shape: (N, 3) where N = number of non-zero voxels

# Extract the corresponding intensities using the mask
intensities = segmented_data[mask]  # Shape: (N,)


# Verify shapes
print(f"Coordinates shape: {coordinates.shape}, Intensities shape: {intensities.shape}")



# %%
coordinates

# %% [markdown]
#  #### 1. Here affine transformation is not considered for the coordinates. So it's may be wrong

# %%

# Create a DataFrame with columns: x, y, z, intensity
df = pd.DataFrame({
    'x': coordinates[:, 0],
    'y': coordinates[:, 1],
    'z': coordinates[:, 2],
    'intensity': intensities
})

# Save to CSV
df.to_csv('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/segmented_intensities_with_coordinates.csv', index=False)



# %%
df

# %% [markdown]
# Plot intensity Histogram 

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['intensity'], bins=100, color='skyblue', alpha=0.7, edgecolor='black',label='Intensity Distribution')
plt.title('Intensity Distribution of Segmented Region')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
# plt.grid(axis='y', alpha=0.3)


threshold = df['intensity'].quantile(0.90)
plt.axvline(threshold, color='red', linestyle='--',linewidth=2, label=f'90th percentile: {threshold:.2f}')
plt.axvline(np.mean(intensities), color='yellow', linestyle='dashed', label=f'Mean: {np.mean(intensities):.2f}')
plt.axvline(np.median(intensities), color='green', linestyle='dashed', label=f'Median: {np.median(intensities):.2f}')
plt.legend()

plt.tight_layout()
# plt.savefig('intensity_histogram.png', dpi=300)  # Save the plot
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/intensity_histogram_1.png', dpi=300)  # Save the plot

plt.show()

# Print key stats
print(f"90th Percentile Threshold: {threshold:.2f}")
print(f"Mean Intensity: {np.mean(intensities):.2f}")
print(f"Median Intensity: {np.median(intensities):.2f} ")
print(f"Standard Deviation: {np.std(intensities):.2f} ")
print(f"min Intensity: {np.min(intensities):.2f} ")
print(f"max Intensity: {np.max(intensities):.2f} ")

# %% [markdown]
# Plot a scatter plot of intensities vs. one spatial dimension:

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['z'], df['intensity'], alpha=0.5, s=2)
plt.title('Intensity Distribution Along Z-axis')
plt.xlabel('Z-coordinate (voxel)')
plt.ylabel('Intensity')
plt.grid(True)
# plt.savefig('intensity_vs_z.png', dpi=300)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/intensity_vs_z_1.png', dpi=300)  # Save the plot

plt.show()

# %% [markdown]
# Plot a 3D scatter plot of the segmented voxels:

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], c=df['intensity'], s=1, alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('3D Distribution of Segmented Voxels')
# plt.savefig('3d_voxels.png', dpi=300)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/3d_voxels_1.png', dpi=300)  # Save the plot

plt.show()

# %%


# %% [markdown]
# #### 2.  Here affine transformation is considered for the coordinates.

# %%
# Add a column of 1s to make homogeneous coordinates (Nx4)
ones = np.ones((coordinates.shape[0], 1))
voxel_coords_homogeneous = np.hstack((coordinates, ones))

# Apply affine transformation
world_coords = voxel_coords_homogeneous @ segmented_img.affine.T  # (Nx4) @ (4x4).T → (Nx4)



# %%
# Drop the last column (homogeneous '1') to get (x, y, z) in mm
world_coords_mm = world_coords[:, :3]    # Remove homogeneous coordinate   # (3D coordinates in mm) 


print(world_coords_mm)

# %%
world_coords_mm.shape

# %%
# Add to DataFrame

# Create a DataFrame with columns: x, y, z, intensity
df_mm = pd.DataFrame({
    'x': world_coords_mm[:, 0],
    'y': world_coords_mm[:, 1],
    'z': world_coords_mm[:, 2],
    'intensity': intensities
})

df_mm.to_csv('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/segmented_intensities_with_coordinates_mm.csv', index=False)


# %%
df_mm

# %% [markdown]
# Plot intensity Histogram

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df_mm['intensity'], bins=100, color='skyblue', alpha=0.7, edgecolor='black',label='Intensity Distribution')
plt.title('Intensity Distribution of Segmented Region')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
# plt.grid(axis='y', alpha=0.3)


threshold = df_mm['intensity'].quantile(0.90)
plt.axvline(threshold, color='red', linestyle='--',linewidth=2, label=f'90th percentile: {threshold:.2f}')


plt.axvline(np.mean(intensities), color='yellow', linestyle='dashed', label=f'Mean: {np.mean(intensities):.2f}')
plt.axvline(np.median(intensities), color='green', linestyle='dashed', label=f'Median: {np.median(intensities):.2f}')
plt.legend()

plt.tight_layout()
# plt.savefig('intensity_histogram.png', dpi=300)  # Save the plot
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/intensity_histogram_2.png', dpi=300)  # Save the plot

plt.show()

# Print key stats
print(f"90th Percentile Threshold: {threshold:.2f}")
print(f"Mean Intensity: {np.mean(intensities):.2f}")
print(f"Median Intensity: {np.median(intensities):.2f} ")
print(f"Standard Deviation: {np.std(intensities):.2f} ")
print(f"min Intensity: {np.min(intensities):.2f} ")
print(f"max Intensity: {np.max(intensities):.2f} ")


# %% [markdown]
# Plot a scatter plot of intensities vs. one spatial dimension:

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df_mm['z'], df['intensity'], alpha=0.5, s=2)
plt.title('Intensity Distribution Along Z-axis')
plt.xlabel('Z-coordinate (voxel)')
plt.ylabel('Intensity')
plt.grid(True)
# plt.savefig('intensity_vs_z.png', dpi=300)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/intensity_vs_z_2.png', dpi=300)  # Save the plot

plt.show()

# %% [markdown]
# Plot a 3D scatter plot of the segmented voxels:

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_mm['x'], df_mm['y'], df_mm['z'], c=df_mm['intensity'], s=1, alpha=0.5)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('3D Distribution of Segmented Voxels')
# plt.savefig('3d_voxels.png', dpi=300)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/3d_voxels_2.png', dpi=300)  # Save the plot


plt.show()

# %%


# %% [markdown]
# ##### Clustering 

# %%
df_mm_cluster = df_mm.copy()

# %%
from sklearn.preprocessing import StandardScaler

X = df_mm_cluster[['x', 'y', 'z', 'intensity']].values  # Extract features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize (mean=0, std=1)

# %% [markdown]
# ##### Optimal  k Clusters (Elbow Method)

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
for k in range(1, 20):  # Test 1 to 20 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/elbow_curve_kmeans.png', dpi=300)  # Save the plot

plt.show()

# %% [markdown]
#  so here optimal cluster will be 5  (Elbow Method)

# %%
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)


# %%
df_mm_cluster['cluster_elbow'] = clusters  

# %%
df_mm_cluster.head(5)

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_mm_cluster['x'], df_mm_cluster['y'], df_mm_cluster['z'], c=df_mm_cluster['cluster_elbow'], cmap='viridis', s=10, alpha=0.6)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.colorbar(scatter, label='Cluster')
plt.title('3D Spatial Clustering of MRI Intensities')
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/kmeans_3d_elbow_5_cluster.png', dpi=300)  # Save the plot

plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster_elbow', y='intensity', data=df_mm_cluster, palette='viridis', hue= "cluster_elbow")
plt.title('Intensity Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Intensity')
plt.grid(True)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/kmeans_box_plot_elbow_5_cluster.png', dpi=300)  # Save the plot

plt.show()

# %% [markdown]
# ##### optimal k using Silhouette Score

# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
sil_scores = []
k_range = range(2, 20)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)


# %%
# Plot silhouette scores
plt.plot(k_range, sil_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")

plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/kmeans_silhouette_plot.png', dpi=300)  # Save the plot

plt.show()

# %%
# Best k
best_k = k_range[np.argmax(sil_scores)]
print(f"Best number of clusters: {best_k}")



# %%

# Final KMeans clustering with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_scaled )
cluster_Silhouette = kmeans.fit_predict(X_scaled )


# %%


# %%
df_mm_cluster['cluster_Silhouette'] = cluster_Silhouette

# %%
df_mm_cluster.head(5)

# %%
df_mm_cluster["cluster_elbow"].unique()

# %%
df_mm_cluster["cluster_Silhouette"].unique()

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_mm_cluster['x'], df_mm_cluster['y'], df_mm_cluster['z'], c=df_mm_cluster['cluster_Silhouette'], cmap='viridis', s=10, alpha=0.6)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.colorbar(scatter, label='cluster_Silhouette')
plt.title('3D Spatial Clustering of MRI Intensities')
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/kmeans_3d_Silhouette_2_cluster.png', dpi=300)  # Save the plot

plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster_Silhouette', y='intensity', data=df_mm_cluster, palette='viridis', hue= "cluster_Silhouette")
plt.title('Intensity Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Intensity')
plt.grid(True)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/kmeans_box_plot_silhouette_2_cluster.png', dpi=300)  # Save the plot

plt.show()

# %%


# %%
segmented_data.shape

# %% [markdown]
# #### To visualize the two clusters in the segmented image

# %%
# import numpy as np
# import nibabel as nib

# # Load the segmented image 
# # segmented_img = nib.load('segmented_region_3.nii.gz')
# # segmented_data = segmented_img.get_fdata()

# # to store cluster labels (same shape as segmented_data)
# cluster_map = np.zeros_like(segmented_data)

# # # Assign cluster labels to voxels
# # for idx, row in df_mm_cluster.iterrows():
# #     x, y, z = int(row['x']), int(row['y']), int(row['z'])
# #     cluster_map[x, y, z] = row['cluster_Silhouette'] + 1  # +1 to avoid 0 (background)



# #  Map the clusters back to the 3D array
# for i, (x, y, z) in enumerate(df_mm_cluster[['x', 'y', 'z']].values.astype(int)):
#     cluster_map[x, y, z] = df_mm_cluster['cluster_Silhouette'].iloc[i]


# %%
# # Save the clustered data as a new NIfTI file
# clustered_img = nib.Nifti1Image(clustered_data, segmented_img.affine, segmented_img.header)
# nib.save(clustered_img, 'segmented_clusters.nii.gz')

# print("Clustered NIfTI file saved as 'segmented_clusters.nii.gz'")

# %%
segmented_img.affine

# %% [markdown]
# use the inverse affine transformation to map the world coordinates (from the CSV) back to voxel indices.

# %%
# Get the affine matrix and its inverse
affine = segmented_img.affine
affine_inv = np.linalg.inv(affine)


# %%
df_mm_cluster

# %%
# Extract world coordinates and convert back to voxel indices
world_coords = df_mm_cluster[['x', 'y', 'z']].values  # World coordinates (in mm)
voxel_coords = np.dot(np.c_[world_coords, np.ones(world_coords.shape[0])], affine_inv.T)[:, :3]  # Convert to voxel indices

# Round and convert to integers
voxel_coords = np.round(voxel_coords).astype(int)



# %%
voxel_coords.shape

# %%
segmented_data.shape

# %%
coordinates

# %%
world_coords_mm

# %%
voxel_coords[:, 0]

# %%
df_mm_cluster_back = pd.DataFrame({
    'x_voxel': voxel_coords[:, 0],
    'y_voxel': voxel_coords[:, 1],
    'z_voxel': voxel_coords[:, 2],
    'intensity': df_mm_cluster['intensity'],
    'cluster_Silhouette': df_mm_cluster['cluster_Silhouette']
})

# %%
df_mm_cluster_back

# %%
df_mm_cluster_back['cluster_Silhouette'] = df_mm_cluster_back['cluster_Silhouette'].replace(0, 2)    # now we have 2 different clusters 1 and 2 . here i have replaced 0 with 2.

# %%
df_mm_cluster_back

# %%
# Create an empty array for the cluster map
cluster_map = np.zeros_like(segmented_data, dtype=np.int32)



# %%
# Map the clusters back to the 3D array
for i, (x, y, z) in enumerate(df_mm_cluster_back[['x_voxel', 'y_voxel', 'z_voxel']].values):
    cluster_map[x, y, z] = df_mm_cluster_back['cluster_Silhouette'].iloc[i]  

# %%
cluster_map.shape

# %%
# cluster_map

# %%
# Save the cluster map as a NIfTI file
cluster_img = nib.Nifti1Image(cluster_map, segmented_img.affine, segmented_img.header)
nib.save(cluster_img, '/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/cluster_2_Silhouette_kmeans.nii.gz')

print("Clustered NIfTI file saved as 'cluster_2_Silhouette_kmeans.nii.gz'")


# %%


# %% [markdown]
# ### GMM 

# %%
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# %%
df_mm_cluster_back

# %%
# Features: x, y, z, intensity
X = df_mm_cluster_back[['x_voxel', 'y_voxel', 'z_voxel', 'intensity']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# Plot BIC and AIC scores
# 

# %%
bic_scores = []
aic_scores = []
k_range = range(1, 20)  # Test 1 to 20 components

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

plt.figure(figsize=(10, 6))
plt.plot(k_range, bic_scores, label='BIC', marker='o')
plt.plot(k_range, aic_scores, label='AIC', marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('BIC and AIC for GMM')
plt.legend()
plt.grid(True)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/gmm_aic_bic_plot.png', dpi=300)  # Save the plot

plt.show()

# %%
#  optimal number of components
optimal_aic = k_range[np.argmin(aic_scores)]  # Index of minimum AIC
optimal_bic = k_range[np.argmin(bic_scores)]  # Index of minimum BIC

print(f"Optimal number of components based on AIC:k= {optimal_aic}")
print(f"Optimal number of components based on BIC:k= {optimal_bic}")

# %%
print(" from the aic and BIC graph it is clear that the optimal number of components is 7. ")

# %%
optimal_components = 7  
gmm = GaussianMixture(n_components=optimal_components, random_state=42)
gmm.fit(X_scaled)

# %%
gmm_clusters = gmm.predict(X_scaled)
df_mm_cluster_back['cluster_gmm'] = gmm_clusters

# %%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_mm_cluster_back['x_voxel'], df_mm_cluster_back['y_voxel'], df_mm_cluster_back['z_voxel'], c=df_mm_cluster_back['cluster_gmm'], cmap='viridis', s=10, alpha=0.6)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.colorbar(scatter, label='Cluster')
plt.title('3D Spatial Clustering with GMM')
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/gmm_3d_7_cluster.png', dpi=300)  # Save the plot

plt.show()

# %%
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster_gmm', y='intensity', data=df_mm_cluster_back, palette='viridis', hue="cluster_gmm")
plt.title('Intensity Distribution by GMM Cluster')
plt.xlabel('Cluster')
plt.ylabel('Intensity')
plt.grid(True)
plt.savefig('/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/gmm_3d_7_cluster.png', dpi=300)  # Save the plot

plt.show()

# %%
df_mm_cluster_back

# %%
df_mm_cluster_back["cluster_gmm"].unique()

# %%
df_mm_cluster_back['cluster_gmm'] = df_mm_cluster_back['cluster_gmm'].replace(0, 16)   

# %%
cluster_map_gmm = np.zeros_like(segmented_data, dtype=np.int32)


# %%
# Map the clusters back to the 3D array
for i, (x, y, z) in enumerate(df_mm_cluster_back[['x_voxel', 'y_voxel', 'z_voxel']].values):
    cluster_map_gmm[x, y, z] = df_mm_cluster_back['cluster_gmm'].iloc[i]  

# %%

cluster_img_gmm = nib.Nifti1Image(cluster_map_gmm, segmented_img.affine, segmented_img.header)
# nib.save(cluster_img_gmm, 'gmm_cluster_labels.nii.gz')
nib.save(cluster_img, '/home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code/cluster_7_gmm.nii.gz')

print("GMM Clustered NIfTI file saved as 'cluster_7_gmm.nii.gz'")

# %%



print("this is the end of the code")