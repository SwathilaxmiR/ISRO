import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.windows import Window

# ---------------------------------------------------
# FUNCTION TO READ SMALL SAR PREVIEW
# ---------------------------------------------------
def read_layer_preview(file_name, title):

    # -----------------------------------------
    # OPEN TIFF
    # -----------------------------------------
    with rasterio.open(file_name) as src:

        print(f"\n{title} Metadata:\n")
        print(src.meta)

        # -----------------------------------------
        # IMAGE SIZE
        # -----------------------------------------
        height = src.height
        width = src.width

        print(f"\nFull Image Shape: ({height}, {width})")

        # -----------------------------------------
        # READ SMALL CENTER WINDOW
        # -----------------------------------------
        window_size = 2000

        row_start = height // 2
        col_start = width // 2

        window = Window(
            col_start,
            row_start,
            window_size,
            window_size
        )

        image = src.read(1, window=window)

    # -----------------------------------------
    # CLEAN VALUES
    # -----------------------------------------
    image = np.nan_to_num(image)

    # -----------------------------------------
    # PRINT STATS
    # -----------------------------------------
    print("\nMin:", np.min(image))
    print("Max:", np.max(image))
    print("Mean:", np.mean(image))

    # -----------------------------------------
    # CONVERT TO dB
    # -----------------------------------------
    image_db = 10 * np.log10(image + 1e-10)

    # -----------------------------------------
    # PERCENTILE CLIP
    # -----------------------------------------
    p2 = np.percentile(image_db, 2)
    p98 = np.percentile(image_db, 98)

    image_db = np.clip(image_db, p2, p98)

    # -----------------------------------------
    # NORMALIZE
    # -----------------------------------------
    image_db = (
        image_db - image_db.min()
    ) / (
        image_db.max() - image_db.min()
    )

    # -----------------------------------------
    # DISPLAY
    # -----------------------------------------
    plt.figure(figsize=(8,8))

    plt.imshow(image_db, cmap='gray')

    plt.title(title)

    plt.colorbar(label="Normalized dB")

    plt.axis("off")

    plt.show()

    return image_db


# ---------------------------------------------------
# READ HH
# ---------------------------------------------------
HH = read_layer_preview("HH.tif", "HH Layer")

# ---------------------------------------------------
# READ HV
# ---------------------------------------------------
HV = read_layer_preview("HV.tif", "HV Layer")

print("\nVisualization Successful!")