import zarr
import numpy as np

# 1. Open the source arrays
ds_start = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h/continuous-24+6h.zarr', mode='r')
ds_mid = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-1/continuous-24+6h.zarr', mode='r')
ds_rest = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-2/continuous-24+6h.zarr', mode='r')

# 2. Define the shape and chunks
# Based on ds_start (12912, 10, 8, 5, 32, 64)
target_shape = ds_start.shape
target_chunks = ds_start.chunks

# 3. Create the new consolidated Zarr array
ds_final = zarr.open('/mnt/data/sonia/cef/results/multivar/continuous-24+6h-final/continuous-24+6h.zarr', 
                     mode='w', 
                     shape=target_shape, 
                     chunks=target_chunks, 
                     dtype=ds_start.dtype)

# 4. Perform the mapping (All indices below follow the "All inclusive" instruction)

# Range 1: ds_start rows 0 to 23 -> Final rows 0 to 23
ds_final[0:24] = ds_start[0:24]

# Range 2: ds_mid rows 0 to 2351 -> Final rows 24 to 2375
# (2352 samples)
ds_final[24:2376] = ds_mid[0:2352]

# Range 3: ds_rest rows 2352 to 12887 -> Final rows 2376 to 12911
# (10536 samples)
ds_final[2376:12912] = ds_rest[2352:12888]

print(f"Consolidated array saved. Final shape: {ds_final.shape}")
