import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

# ---------------------------------------------------
# FILE PATH
# ---------------------------------------------------
file_path = r"data\nisar1.h5"

# ---------------------------------------------------
# OPEN HDF5
# ---------------------------------------------------
f = h5py.File(file_path, "r")

print("\nAVAILABLE DATASETS:\n")

def print_name(name):
    print(name)

f.visit(print_name)

# ---------------------------------------------------
# POLARIZATION LAYERS
# ---------------------------------------------------
layers = {
    "HH": "/science/LSAR/GCOV/grids/frequencyA/HHHH",
    "HV": "/science/LSAR/GCOV/grids/frequencyA/HVHV",
    "VV": "/science/LSAR/GCOV/grids/frequencyA/VVVV"
}

# ---------------------------------------------------
# READ COORDINATES
# ---------------------------------------------------
x_coords = f["/science/LSAR/GCOV/grids/frequencyA/xCoordinates"][:]
y_coords = f["/science/LSAR/GCOV/grids/frequencyA/yCoordinates"][:]

# ---------------------------------------------------
# PIXEL SPACING
# ---------------------------------------------------
pixel_width = x_coords[1] - x_coords[0]
pixel_height = y_coords[1] - y_coords[0]

# ---------------------------------------------------
# GEO TRANSFORM
# ---------------------------------------------------
transform = from_origin(
    x_coords.min(),
    y_coords.max(),
    pixel_width,
    abs(pixel_height)
)

# ---------------------------------------------------
# CRS   
# ---------------------------------------------------
try:
    epsg = int(f["/science/LSAR/GCOV/grids/frequencyA/projection"][:])
    crs = f"EPSG:{epsg}"
except:
    crs = "EPSG:4326"

print("\nCRS:", crs)

# ---------------------------------------------------
# CONVERT EACH LAYER
# ---------------------------------------------------
for layer_name, dataset_path in layers.items():

    print(f"\nProcessing {layer_name}...")

    # Read data
    data = f[dataset_path][:]

    # Convert complex → magnitude
    data = np.abs(data).astype(np.float32)

    # Output filename
    output_tiff = f"{layer_name}.tif"

    # Save GeoTIFF
    with rasterio.open(
        output_tiff,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:

        dst.write(data, 1)

    print(f"{output_tiff} created successfully!")

print("\nALL LAYERS CONVERTED!")