# mkdir 1.0 && tar -xzf cef_sample_multivar_6292903_1_0.tar.gz -C 1.0 --strip-components 3
# mkdir 1.1 && tar -xzf cef_sample_multivar_6292903_1_1.tar.gz -C 1.1 --strip-components 3
# mkdir 1.2 && tar -xzf cef_sample_multivar_6292903_1_2.tar.gz -C 1.2 --strip-components 3
# mkdir 1.3 && tar -xzf cef_sample_multivar_6292903_1_3.tar.gz -C 1.3 --strip-components 3
# mkdir 1.4 && tar -xzf cef_sample_multivar_6292903_1_4.tar.gz -C 1.4 --strip-components 3
# mkdir 1.5 && tar -xzf cef_sample_multivar_6292903_1_5.tar.gz -C 1.5 --strip-components 3
# mkdir 1.6 && tar -xzf cef_sample_multivar_6292903_1_6.tar.gz -C 1.6 --strip-components 3
# mkdir 1.7 && tar -xzf cef_sample_multivar_6292903_1_7.tar.gz -C 1.7 --strip-components 3
# mkdir 1.8 && tar -xzf cef_sample_multivar_6292903_1_8.tar.gz -C 1.8 --strip-components 3
# mkdir 1.9 && tar -xzf cef_sample_multivar_6292903_1_9.tar.gz -C 1.9 --strip-components 3
# mkdir 1.10 && tar -xzf cef_sample_multivar_6292903_1_10.tar.gz -C 1.10 --strip-components 3
# mkdir 1.11 && tar -xzf cef_sample_multivar_6292903_1_11.tar.gz -C 1.11 --strip-components 3
# mkdir 1.12 && tar -xzf cef_sample_multivar_6292903_1_12.tar.gz -C 1.12 --strip-components 3
# mkdir 1.13 && tar -xzf cef_sample_multivar_6292903_1_13.tar.gz -C 1.13 --strip-components 3
# mkdir 1.14 && tar -xzf cef_sample_multivar_6292903_1_14.tar.gz -C 1.14 --strip-components 3

import zarr
import numpy as np
import os
from tqdm import tqdm

basedir = '/mnt/data/sonia/cef/results/multivar/3'
subdirs = [f'1.{i}' for i in range(15)]

dss = [zarr.open(os.path.join(basedir, subdir, 'continuous-24+6h.zarr'), mode='r') for subdir in subdirs]

# Define the shape and chunks
# Based on (13145, 10, 8, 5, 32, 64)
target_shape = dss[0].shape
target_chunks = dss[0].chunks

# Create the new consolidated Zarr array
ds_final = zarr.open(os.path.join(basedir, 'final/continuous-24+6h.zarr'), 
                     mode='w', 
                     shape=target_shape, 
                     chunks=target_chunks, 
                     dtype=dss[0].dtype)

for i, ds in enumerate(tqdm(dss)):
    start = 877 * i 
    end = 877 * (i+1)
    if end > target_shape[0]:
        end = target_shape[0]
    ds_final[start:end] = ds[start:end]
    

# # 1. Open the source arrays
# ds_0 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-0/continuous-24+6h.zarr', mode='r')
# ds_1 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-1/continuous-24+6h.zarr', mode='r')
# ds_2 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-2/continuous-24+6h.zarr', mode='r')
# ds_3 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-3/continuous-24+6h.zarr', mode='r')

# # 2. Define the shape and chunks
# # Based on ds_3 (13145, 10, 8, 5, 32, 64)
# target_shape = ds_3.shape
# target_chunks = ds_3.chunks

# # 3. Create the new consolidated Zarr array
# ds_final = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-final/continuous-24+6h.zarr', 
#                      mode='w', 
#                      shape=target_shape, 
#                      chunks=target_chunks, 
#                      dtype=ds_3.dtype)

# # 4. Perform the mapping

# # Range 1: ds_0 rows 0 to 23 -> Final rows 0 to 23
# ds_final[0:24] = ds_0[0:24]

# # Range 2: ds_1 rows 0 to 2351 -> Final rows 24 to 2375
# # (2352 samples)
# ds_final[24:2376] = ds_1[0:2352]

# # Range 3: ds_2 rows 2352 to 12887 -> Final rows 2376 to 12911
# # (10536 samples)
# ds_final[2376:12888] = ds_2[2352:12864]

# # Range 4: ds_3 rows 12912 to 13144 -> Final rows 12912 to 13144
# # (2333 samples)
# ds_final[12888:13145] = ds_3[12888:13145]


# print(f"Consolidated array saved. Final shape: {ds_final.shape}")
