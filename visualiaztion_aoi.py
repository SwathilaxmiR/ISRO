import rasterio
import numpy as np
import matplotlib.pyplot as plt

def prepare_image(file_path):

    with rasterio.open(file_path) as src:
        img = src.read(1)

    img = np.nan_to_num(img)

    img = 10 * np.log10(img + 1e-10)

    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)

    img = np.clip(img, p2, p98)

    img = (img - img.min()) / (img.max() - img.min())

    return img


HH = prepare_image(
    r"HH_tiles\tile_r10000_c12000.tif"
)

HV = prepare_image(
    r"HV_tiles\tile_r10000_c12000.tif"
)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(HH, cmap="gray")
plt.title("HH")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(HV, cmap="gray")
plt.title("HV")
plt.axis("off")

plt.show()