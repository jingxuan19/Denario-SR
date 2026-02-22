# filename: check_hdf5_structure.py
import h5py

# Open the HDF5 file in read mode
with h5py.File("data/synthetic_data.h5", "r") as f:
    # Print available datasets/groups
    print("Available datasets/groups in HDF5 file:")
    for key in f.keys():
        print(key)