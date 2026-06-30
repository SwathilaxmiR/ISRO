import io
import time
import base64
import zipfile
import os
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from skimage.transform import resize

app = Flask(__name__)
CORS(app)

MODEL_DIR = "d:/ISRO/Proj"
print("Loading ML models...")
unet_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_u-net_e04.h5"), compile=False, safe_mode=False)
cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_cnn_e04.h5"), compile=False, safe_mode=False)
vision_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_vision_e04.h5"), compile=False, safe_mode=False)
print("Models loaded successfully.")

def get_band_config(band):
    configs = {
        "L": {"lower": -7.0, "upper": 3.5, "w": 10.1, "l": 19.6, "c": 20.8,
              "freq_ghz": "1.0–2.0 GHz", "wavelength_cm": "15–30 cm",
              "penetration": "Deep canopy, soil, subsurface",
              "application": "Soil moisture, geology, flood mapping"},
        "C": {"lower": -15.0, "upper": 5.0, "w": 8.0, "l": 15.0, "c": 18.0,
              "freq_ghz": "4.0–8.0 GHz", "wavelength_cm": "3.75–7.5 cm",
              "penetration": "Shallow canopy, vegetation surface",
              "application": "Crop monitoring, ocean surface, sea ice"},
        "S": {"lower": -10.0, "upper": 0.0, "w": 9.5, "l": 17.0, "c": 19.5,
              "freq_ghz": "2.0–4.0 GHz", "wavelength_cm": "7.5–15 cm",
              "penetration": "Medium penetration, upper canopy",
              "application": "Agriculture, urban mapping, precipitation"}
    }
    return configs.get(band, configs["L"])

def np_to_base64(img_array):
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/api/process", methods=["POST"])
def process():
    band = request.form.get("band", "L")
    hh_input = request.form.get("hh_path")
    hv_input = request.form.get("hv_path")
    
    if not hh_input or not hv_input:
        return jsonify({"error": "Missing HH or HV paths"}), 400

    t0 = time.time()
    
    # Helper to resolve a path to a list of .tif files
    def resolve_paths(input_path):
        resolved = []
        if os.path.isfile(input_path) and input_path.lower().endswith(('.tif', '.tiff')):
            resolved.append(input_path)
        elif os.path.isdir(input_path):
            for f in os.listdir(input_path):
                if f.lower().endswith(('.tif', '.tiff')):
                    resolved.append(os.path.join(input_path, f))
        return resolved

    hh_paths = resolve_paths(hh_input)
    hv_paths = resolve_paths(hv_input)

    if not hh_paths:
        return jsonify({"error": f"No HH .tif files found at {hh_input}."}), 400
        
    if not hv_paths:
        hv_paths = hh_paths

    try:
        from rasterio.enums import Resampling
        from rasterio.merge import merge
        import rasterio

        # Load or Stitch HH
        if len(hh_paths) == 1:
            with rasterio.open(hh_paths[0]) as ds:
                profile = ds.profile
                transform = ds.transform
                crs = str(ds.crs) if ds.crs else "Not Defined"
                bounds = ds.bounds
                dtype = str(ds.dtypes[0])
                band_count = ds.count
                
                scale = min(1024 / ds.height, 1024 / ds.width)
                if scale >= 1:
                    out_shape = (ds.height, ds.width)
                else:
                    out_shape = (int(ds.height * scale), int(ds.width * scale))
                    
                hh = ds.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)
                
                extracted_hv_from_band2 = False
                if band_count >= 2 and len(hv_paths) == 1 and hh_paths[0] == hv_paths[0]:
                    hv = ds.read(2, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)
                    extracted_hv_from_band2 = True
                    
            if not extracted_hv_from_band2:
                with rasterio.open(hv_paths[0]) as ds:
                    hv = ds.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)
        else:
            # Multiple tiles -> Mosaicking
            hh_datasets = [rasterio.open(p) for p in hh_paths]
            ds_0 = hh_datasets[0]
            profile = ds_0.profile
            crs = str(ds_0.crs) if ds_0.crs else "Not Defined"
            dtype = str(ds_0.dtypes[0])
            band_count = ds_0.count
            
            hh_mosaic, transform = merge(hh_datasets)
            hh_array = hh_mosaic[0].astype(np.float32)
            bounds = None
            for ds in hh_datasets: ds.close()
            
            # Decimate in memory after mosaic
            scale = min(1024 / hh_array.shape[0], 1024 / hh_array.shape[1])
            if scale < 1:
                out_shape = (int(hh_array.shape[0] * scale), int(hh_array.shape[1] * scale))
                hh = resize(hh_array, out_shape, anti_aliasing=True, preserve_range=True).astype(np.float32)
            else:
                hh = hh_array
                
            extracted_hv_from_band2 = False
            
            # HV Mosaicking
            hv_datasets = [rasterio.open(p) for p in hv_paths]
            hv_mosaic, _ = merge(hv_datasets)
            hv_array = hv_mosaic[0].astype(np.float32)
            for ds in hv_datasets: ds.close()
            
            if scale < 1:
                hv = resize(hv_array, out_shape, anti_aliasing=True, preserve_range=True).astype(np.float32)
            else:
                hv = hv_array

    except Exception as e:
        return jsonify({"error": "Failed to read or stitch TIF tensors: " + str(e)}), 500

    # Sanitize NaNs which cause histogram crashes (e.g. from nodata regions)
    hh = np.nan_to_num(hh, nan=1e-9)
    hv = np.nan_to_num(hv, nan=1e-9)

    height, width = hh.shape

    # 3. Finalize polarizations (fallback synthesis if HV genuinely missing)
    np.random.seed(42)
    if 'hv' not in locals() or (len(hh_paths) == 1 and not extracted_hv_from_band2):
        hv = hh * 0.3 + (np.random.rand(height, width) * 10).astype(np.float32)
        synth_method = "hv = hh × 0.3 + 𝒩(0, 10)"
        synth_desc = "Missing HV polarization synthesized from HH."
    else:
        synth_method = "Dual-polarization explicitly extracted from local paths."
        synth_desc = f"Extracted HH and HV arrays directly from {len(hh_paths)} physical TIF file(s)."
        # Ensure dimensions match if they came from different files
        if hh.shape != hv.shape:
            min_h = min(hh.shape[0], hv.shape[0])
            min_w = min(hh.shape[1], hv.shape[1])
            hh = hh[:min_h, :min_w]
            hv = hv[:min_h, :min_w]
            height, width = hh.shape

    # --- Stage 1: Raw visualization (using HH) ---
    raw_min, raw_max = float(np.min(hh)), float(np.max(hh))
    raw_vis = np.clip((hh - raw_min) / (raw_max - raw_min + 1e-9) * 255, 0, 255).astype(np.uint8)

    # --- Stage 2: dB Stretch ---
    config = get_band_config(band)
    lower, upper = config["lower"], config["upper"]

    def to_db(arr):
        arr = np.where(arr <= 0, 1e-9, arr)
        return 10.0 * np.log10(arr)

    def stretch(db, lo, hi):
        s = (db - lo) / (hi - lo) * 255.0
        return np.clip(s, 0, 255).astype(np.uint8)

    hh_db = to_db(hh)
    hv_db = to_db(hv)
    ratio_db = to_db(hh / np.where(hv <= 0, 1e-9, hv))
    diff_db = to_db(np.abs(hh - hv) + 1e-9)

    composite_db = (hh_db + hv_db + ratio_db + diff_db) / 4.0
    stretched = stretch(composite_db, lower, upper)

    # --- Stage 3: K-Means Threshold Mask ---
    db_img = 10 * np.log10(np.where(stretched > 0, stretched, 1e-9))
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    water_mask   = db_img <= config["w"]
    land_mask    = (db_img > config["w"]) & (db_img <= config["l"])
    crop_mask    = (db_img > config["l"]) & (db_img <= config["c"])
    mtn_mask     = db_img > config["c"]

    mask[water_mask]  = [41,  128, 185]
    mask[land_mask]   = [39,  174, 96]
    mask[crop_mask]   = [243, 156, 18]
    mask[mtn_mask]    = [192, 57,  43]

    # --- Stage 4: ML Models Inference ---
    # Replicate train_unet.py preprocessing EXACTLY to fix misclassifications
    ratio_raw = hh / np.where(hv <= 0, 1e-9, hv)
    diff_raw = hh - hv
    hv_raw = np.where(hv <= 0, 1e-9, hv)
    
    ratio_db_ml = 10 * np.log10(np.where(ratio_raw <= 0, 1e-9, ratio_raw))
    diff_db_ml  = 10 * np.log10(np.where(diff_raw <= 0, 1e-9, diff_raw))
    hv_db_ml    = 10 * np.log10(hv_raw)
    
    # The loaded models are _e04.h5 (C-band), which were trained with stretch bounds -20.0 and 5.0
    ratio_str = stretch(ratio_db_ml, -20.0, 5.0)
    diff_str  = stretch(diff_db_ml, -20.0, 5.0)
    hv_str_ml = stretch(hv_db_ml, -20.0, 5.0)
    
    # Calculate composite (0-255)
    gray_composite_ml = 0.33 * hv_str_ml + 0.33 * ratio_str + 0.33 * diff_str
    
    # 1. High-Resolution Patch-Based Inference (U-Net & CNN)
    def get_fcn_mask(model, composite_img, out_h, out_w):
        patch_size = 256
        c_mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # Pad image to be a multiple of patch_size
        pad_h = (patch_size - out_h % patch_size) % patch_size
        pad_w = (patch_size - out_w % patch_size) % patch_size
        padded_img = np.pad(composite_img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        for y in range(0, padded_img.shape[0], patch_size):
            for x in range(0, padded_img.shape[1], patch_size):
                patch = padded_img[y:y+patch_size, x:x+patch_size]
                patch_tensor = np.expand_dims(patch, axis=(0, -1))
                
                pred_prob = model.predict(patch_tensor, verbose=0)
                pred_class = np.argmax(pred_prob, axis=-1)[0]
                
                p_mask = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                p_mask[pred_class == 0] = [41,  128, 185] # Water
                p_mask[pred_class == 1] = [39,  174, 96]  # Land
                p_mask[pred_class == 2] = [243, 156, 18]  # Crop
                p_mask[pred_class == 3] = [192, 57,  43]  # Mountain
                
                valid_h = min(patch_size, out_h - y)
                valid_w = min(patch_size, out_w - x)
                
                if valid_h > 0 and valid_w > 0:
                    c_mask[y:y+valid_h, x:x+valid_w] = p_mask[:valid_h, :valid_w]
                    
        return c_mask

    unet_mask = get_fcn_mask(unet_model, gray_composite_ml, height, width)
    cnn_mask = get_fcn_mask(cnn_model, gray_composite_ml, height, width)
    
    # 2. Vision Transformer processing - requires exactly 256x256
    vit_composite = resize(gray_composite_ml, (256, 256), anti_aliasing=True, mode='reflect', preserve_range=True)
    vit_tensor = np.expand_dims(vit_composite, axis=(0, -1))
    
    vit_prob = vision_model.predict(vit_tensor, verbose=0)
    vit_class = np.argmax(vit_prob, axis=-1)[0]
    
    v_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    v_mask[vit_class == 0] = [41,  128, 185]
    v_mask[vit_class == 1] = [39,  174, 96]
    v_mask[vit_class == 2] = [243, 156, 18]
    v_mask[vit_class == 3] = [192, 57,  43]
    
    vision_mask = resize(v_mask, (height, width), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)


    process_ms = int((time.time() - t0) * 1000)
    total_px = height * width

    # --- Statistics ---
    raw_flat = hh.flatten()
    composite_flat = composite_db.flatten()

    hist_counts, hist_bins = np.histogram(composite_flat, bins=24)
    histogram = [{"bin": round(float(hist_bins[i]), 2), "count": int(hist_counts[i])} for i in range(len(hist_counts))]

    raw_hist_counts, raw_hist_bins = np.histogram(raw_flat[raw_flat > 0], bins=24)
    raw_histogram = [{"bin": round(float(raw_hist_bins[i]), 1), "count": int(raw_hist_counts[i])} for i in range(len(raw_hist_counts))]

    water_pct   = round(float(np.sum(water_mask) / total_px) * 100, 2)
    land_pct    = round(float(np.sum(land_mask)  / total_px) * 100, 2)
    crop_pct    = round(float(np.sum(crop_mask)  / total_px) * 100, 2)
    mtn_pct     = round(float(np.sum(mtn_mask)   / total_px) * 100, 2)

    snr         = round(float(np.mean(raw_vis) / (np.std(raw_vis) + 1e-9)), 4)
    entropy     = round(float(-np.sum(np.where(raw_flat > 0,
                    raw_flat / (np.sum(raw_flat) + 1e-9) * np.log2(raw_flat / (np.sum(raw_flat) + 1e-9) + 1e-12), 0))), 4)

    db_valid = composite_flat[np.isfinite(composite_flat)]

    result = {
        "images": {
            "raw":      np_to_base64(raw_vis),
            "stretched": np_to_base64(stretched),
            "mask":     np_to_base64(mask),
            "unet":     np_to_base64(unet_mask),
            "cnn":      np_to_base64(cnn_mask),
            "vision":   np_to_base64(vision_mask)
        },
        "file_metadata": {
            "filename":    "Mosaicked" if len(hh_paths) > 1 else os.path.basename(hh_paths[0]),
            "size_mb":     round(sum(os.path.getsize(p) for p in hh_paths + hv_paths) / (1024 * 1024), 3),
            "dtype":       dtype,
            "band_count":  band_count,
            "resolution":  str(width) + " x " + str(height),
            "total_pixels": total_px,
            "crs":         crs,
            "bounds": {
                "left":   round(bounds.left,  6),
                "bottom": round(bounds.bottom, 6),
                "right":  round(bounds.right,  6),
                "top":    round(bounds.top,    6)
            },
            "transform": [round(float(v), 8) for v in [transform.a, transform.b, transform.c,
                                                         transform.d, transform.e, transform.f]]
        },
        "raw_array_stats": {
            "min":    round(float(raw_min), 4),
            "max":    round(float(raw_max), 4),
            "mean":   round(float(np.mean(hh)), 4),
            "std":    round(float(np.std(hh)), 4),
            "variance": round(float(np.var(hh)), 4),
            "median": round(float(np.median(hh)), 4),
            "p5":     round(float(np.percentile(hh, 5)), 4),
            "p95":    round(float(np.percentile(hh, 95)), 4)
        },
        "db_stats": {
            "mean":   round(float(np.mean(db_valid)), 4),
            "std":    round(float(np.std(db_valid)), 4),
            "min":    round(float(np.min(db_valid)), 4),
            "max":    round(float(np.max(db_valid)), 4),
            "range":  round(float(np.max(db_valid) - np.min(db_valid)), 4)
        },
        "band_config": {
            "band": band,
            "lower_db": lower,
            "upper_db": upper,
            "water_threshold_db": config["w"],
            "land_threshold_db":  config["l"],
            "crop_threshold_db":  config["c"],
            "freq_ghz":    config["freq_ghz"],
            "wavelength_cm": config["wavelength_cm"],
            "penetration": config["penetration"],
            "application": config["application"]
        },
        "classification": {
            "water_pct":    water_pct,
            "land_pct":     land_pct,
            "crop_pct":     crop_pct,
            "mountain_pct": mtn_pct,
            "water_px":     int(np.sum(water_mask)),
            "land_px":      int(np.sum(land_mask)),
            "crop_px":      int(np.sum(crop_mask)),
            "mountain_px":  int(np.sum(mtn_mask))
        },
        "quality_metrics": {
            "snr":          snr,
            "entropy_bits": round(abs(entropy), 4),
            "process_ms":   process_ms,
            "dynamic_range_db": round(float(raw_max - raw_min), 4)
        },
        "histogram":     histogram,
        "raw_histogram": raw_histogram,
        "synth": {
            "method": synth_method,
            "desc": synth_desc,
            "raw_label": "HH/HV"
        }
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
