import zarr
import numpy as np

# 1. Open the source arrays
ds_0 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-0/continuous-24+6h.zarr', mode='r')
ds_1 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-1/continuous-24+6h.zarr', mode='r')
ds_2 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-2/continuous-24+6h.zarr', mode='r')
ds_3 = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-3/continuous-24+6h.zarr', mode='r')

# 2. Define the shape and chunks
# Based on ds_3 (13145, 10, 8, 5, 32, 64)
target_shape = ds_3.shape
target_chunks = ds_3.chunks

# 3. Create the new consolidated Zarr array
ds_final = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-final/continuous-24+6h.zarr', 
                     mode='w', 
                     shape=target_shape, 
                     chunks=target_chunks, 
                     dtype=ds_3.dtype)

# 4. Perform the mapping

# Range 1: ds_0 rows 0 to 23 -> Final rows 0 to 23
ds_final[0:24] = ds_0[0:24]

# Range 2: ds_1 rows 0 to 2351 -> Final rows 24 to 2375
# (2352 samples)
ds_final[24:2376] = ds_1[0:2352]

# Range 3: ds_2 rows 2352 to 12887 -> Final rows 2376 to 12911
# (10536 samples)
ds_final[2376:12888] = ds_2[2352:12864]

# Range 4: ds_3 rows 12912 to 13144 -> Final rows 12912 to 13144
# (2333 samples)
ds_final[12888:13145] = ds_3[12888:13145]


print(f"Consolidated array saved. Final shape: {ds_final.shape}")
