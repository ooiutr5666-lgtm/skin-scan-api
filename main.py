import io
import os
import glob
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# พยายามนำเข้าโมดูลประมวลผลภาพและสกัด Features ตามโค้ด Colab
try:
    import cv2
    from scipy.stats import skew
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.measure import label, regionprops
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import features
    FEATURES_MODULE_AVAILABLE = True
except ImportError:
    FEATURES_MODULE_AVAILABLE = False

app = FastAPI(
    title="Skin Lesion Classifier API (Skin Cancer: Malignant vs. Benign)",
    description="API สำหรับคัดกรองมะเร็งผิวหนังด้วยโมเดล ABCD Handcrafted Features + SVM/Deep Learning (ISIC/Kaggle)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['benign', 'malignant'] # 0 = benign (ไฝปกติ), 1 = malignant (มะเร็งผิวหนัง)

# รายชื่อไฟล์โมเดลที่บันทึกจาก Google Colab (ให้ bundle.joblib มีลำดับความสำคัญสูงสุด)
MODEL_FILENAMES = [
    'bundle.joblib',
    'skin_cancer_pipeline.joblib',
    'skin_cancer_pipeline (2).joblib',
    'skin_cancer_pipeline (1).joblib',
    'model.joblib',
    'model (2).joblib',
    'model (1).joblib',
    'skin_cancer_model.joblib',
    'model.pkl',
    'pipeline.pkl'
]

# รายชื่อ 58 ฟีเจอร์มาตรฐานตาม Colab (เรียงตามลำดับการสกัดเป๊ะ 100%)
DEFAULT_FEAT_COLS = [
    # A – Asymmetry (3)
    'A_asym_horizontal', 'A_asym_vertical', 'A_asym_mean',
    # B & D – Border & Diameter (10)
    'B_compactness', 'B_solidity', 'B_convexity', 'B_eccentricity', 'B_extent',
    'D_area', 'D_perimeter', 'D_equiv_diameter', 'D_major_axis', 'D_minor_axis',
    # C – Color RGB (9)
    'C_R_mean', 'C_R_std', 'C_R_skew',
    'C_G_mean', 'C_G_std', 'C_G_skew',
    'C_B_mean', 'C_B_std', 'C_B_skew',
    # C – Color HSV (9)
    'C_H_mean', 'C_H_std', 'C_H_skew',
    'C_S_mean', 'C_S_std', 'C_S_skew',
    'C_V_mean', 'C_V_std', 'C_V_skew',
    # C – Color Lab (9)
    'C_L_mean', 'C_L_std', 'C_L_skew',
    'C_a_mean', 'C_a_std', 'C_a_skew',
    'C_b_mean', 'C_b_std', 'C_b_skew',
    # C – Contrast & Color count (2)
    'C_contrast_to_skin', 'C_n_colors',
    # Texture – GLCM (6)
    'T_glcm_contrast', 'T_glcm_dissimilarity', 'T_glcm_homogeneity',
    'T_glcm_energy', 'T_glcm_correlation', 'T_glcm_ASM',
    # Texture – LBP (10)
    'T_lbp_0', 'T_lbp_1', 'T_lbp_2', 'T_lbp_3', 'T_lbp_4',
    'T_lbp_5', 'T_lbp_6', 'T_lbp_7', 'T_lbp_8', 'T_lbp_9'
]

loaded_pipeline = None
loaded_threshold = 0.22  # เกณฑ์คัดกรองมะเร็งผิวหนัง (Recall >= 98%) ตาม Colab v4 (t_lo)
loaded_feat_cols = None
loaded_classes = [0, 1]
loaded_ood = None  # ตัวตรวจภาพนอกขอบเขต (Mahalanobis) จาก Colab ถ้ามี
loaded_model_name = None
loaded_model_size = 0
loaded_cnn = None
loaded_cnn_type = None  # 'onnx', 'tflite', or 'keras'
loaded_w_cnn = 0.0
loaded_t_lo = 0.22
loaded_t_hi = 0.51

loaded_cnn_error = None

def get_trained_cnn():
    global loaded_cnn, loaded_cnn_type, loaded_cnn_error
    if loaded_cnn is not None:
        return loaded_cnn
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cnn_candidates = [
        (os.path.join(base_dir, 'cnn.onnx'), 'onnx'),
        ('cnn.onnx', 'onnx'),
        (os.path.join(base_dir, 'model.onnx'), 'onnx'),
        ('model.onnx', 'onnx'),
        (os.path.join(base_dir, 'cnn.tflite'), 'tflite'),
        ('cnn.tflite', 'tflite'),
        (os.path.join(base_dir, 'cnn.keras'), 'keras'),
        ('cnn.keras', 'keras'),
    ]
    for f in glob.glob(os.path.join(base_dir, "*.onnx")) + glob.glob("*.onnx"):
        if (f, 'onnx') not in cnn_candidates:
            cnn_candidates.append((f, 'onnx'))

    for cpath, ctype in cnn_candidates:
        if os.path.exists(cpath):
            try:
                if ctype == 'onnx':
                    import onnxruntime as ort
                    loaded_cnn = ort.InferenceSession(cpath, providers=['CPUExecutionProvider'])
                    loaded_cnn_type = 'onnx'
                    loaded_cnn_error = None
                    print(f"[OK] CNN Loaded via ONNX Runtime: {cpath}")
                    return loaded_cnn
                elif ctype == 'tflite':
                    try:
                        import tflite_runtime.interpreter as tflite
                    except ImportError:
                        import tensorflow.lite as tflite
                    interpreter = tflite.Interpreter(model_path=cpath)
                    interpreter.allocate_tensors()
                    loaded_cnn = interpreter
                    loaded_cnn_type = 'tflite'
                    loaded_cnn_error = None
                    print(f"[OK] CNN Loaded via TFLite: {cpath}")
                    return loaded_cnn
                elif ctype == 'keras':
                    try:
                        import tensorflow as tf
                        loaded_cnn = tf.keras.models.load_model(cpath, compile=False)
                        loaded_cnn_type = 'keras'
                        loaded_cnn_error = None
                        print(f"[OK] CNN Loaded via Keras/TensorFlow: {cpath}")
                        return loaded_cnn
                    except ImportError:
                        import keras
                        loaded_cnn = keras.models.load_model(cpath, compile=False)
                        loaded_cnn_type = 'keras'
                        loaded_cnn_error = None
                        print(f"[OK] CNN Loaded via Keras: {cpath}")
                        return loaded_cnn
            except Exception as e:
                loaded_cnn_error = f"{type(e).__name__}: {str(e)}"
                print(f"[WARN] Failed to load CNN {cpath}: {e}")
    if loaded_cnn_error is None:
        loaded_cnn_error = f"No CNN file found in {base_dir} (files={os.listdir(base_dir)})"
    return None

def predict_cnn_prob(img_rgb):
    """ทำนายความน่าจะเป็นของมะเร็งผิวหนังด้วยโมเดล CNN (EfficientNetB0)"""
    cnn = get_trained_cnn()
    if cnn is None:
        return None
    try:
        img_resized = cv2.resize(img_rgb, (224, 224))
        x = np.expand_dims(img_resized.astype(np.float32), axis=0)
        
        if loaded_cnn_type == 'onnx':
            input_name = cnn.get_inputs()[0].name
            preds = cnn.run(None, {input_name: x})[0]
            return float(np.squeeze(preds))
        elif loaded_cnn_type == 'tflite':
            input_details = cnn.get_input_details()
            output_details = cnn.get_output_details()
            cnn.set_tensor(input_details[0]['index'], x)
            cnn.invoke()
            preds = cnn.get_tensor(output_details[0]['index'])
            return float(preds[0][0])
        elif loaded_cnn_type == 'keras':
            try:
                preds = loaded_cnn(x, training=False)
                if hasattr(preds, 'numpy'):
                    return float(preds.numpy()[0][0])
                return float(preds[0][0])
            except Exception:
                preds = loaded_cnn.predict(x, verbose=0)
                return float(preds[0][0])
    except Exception as e:
        print(f"[WARN] CNN prediction failed: {e}")
        return None

def get_trained_pipeline():
    global loaded_pipeline, loaded_threshold, loaded_feat_cols, loaded_classes, loaded_ood, loaded_model_name, loaded_model_size, loaded_w_cnn, loaded_t_lo, loaded_t_hi
    if loaded_pipeline is not None:
        return loaded_pipeline
    if JOBLIB_AVAILABLE:
        candidates = list(MODEL_FILENAMES)
        for f in glob.glob("*.joblib") + glob.glob("*.pkl"):
            if f not in candidates:
                candidates.append(f)

        for fname in candidates:
            if os.path.exists(fname):
                try:
                    obj = joblib.load(fname)
                    if isinstance(obj, dict):
                        loaded_pipeline = obj.get('model') or obj.get('hand') or obj.get('pipeline') or obj.get('clf')
                        if loaded_pipeline is None:
                            for v in obj.values():
                                if hasattr(v, 'predict') or hasattr(v, 'predict_proba'):
                                    loaded_pipeline = v
                                    break
                        if loaded_pipeline is None:
                            loaded_pipeline = obj

                        loaded_t_hi = float(obj.get('t_hi', 0.51))
                        loaded_t_lo = float(obj.get('t_lo', 0.22))
                        # ใช้ t_lo เป็นเกณฑ์คัดกรองหลัก (Screening Threshold) เพื่อให้ Recall >= 98% ไม่หลุดมะเร็ง
                        loaded_threshold = float(loaded_t_lo)
                        loaded_w_cnn = float(obj.get('w_cnn', 0.7 if get_trained_cnn() is not None else 0.0))
                        loaded_feat_cols = obj.get('feat_cols', None)
                        loaded_ood = obj.get('ood', None)
                        if 'classes' in obj:
                            loaded_classes = list(obj['classes'])
                    else:
                        loaded_pipeline = obj

                    loaded_model_name = fname
                    loaded_model_size = os.path.getsize(fname)

                    if hasattr(loaded_pipeline, 'classes_'):
                        loaded_classes = list(loaded_pipeline.classes_)
                    elif hasattr(loaded_pipeline, 'named_steps') and 'clf' in loaded_pipeline.named_steps:
                        clf = loaded_pipeline.named_steps['clf']
                        if hasattr(clf, 'classes_'):
                            loaded_classes = list(clf.classes_)

                    print(f"[OK] Model loaded from: {fname} (size={loaded_model_size}, threshold={loaded_threshold}, classes={loaded_classes}, w_cnn={loaded_w_cnn})")
                    return loaded_pipeline
                except Exception as e:
                    print(f"[WARN] Failed to load model from {fname}: {e}")
    return None

# =============================================================
# 1) Preprocessing: ลบขน (DullRazor แบบย่อ) ตาม Colab
# =============================================================
def remove_hair(img_rgb):
    if not OPENCV_AVAILABLE:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_rgb, hair_mask, 3, cv2.INPAINT_TELEA)

# =============================================================
# 2) Segmentation: ชดเชยแสง (flat-field) + Otsu ตาม Colab เป๊ะ
# =============================================================
def _otsu_pick(M, h, w):
    """Otsu บนแผนที่ความเข้ม แล้วเลือก component ที่ใหญ่และอยู่ใกล้กลางภาพ (ตาม Colab ข้อ 2)"""
    if not OPENCV_AVAILABLE:
        return None
    M = cv2.GaussianBlur(M, (5, 5), 0)
    M8 = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(M8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    lab = label(mask > 0)
    candidates = []
    for r in regionprops(lab):
        if r.area < (0.002 * h * w):
            continue
        cy, cx = r.centroid
        dist = np.hypot(cy - h / 2, cx - w / 2) / (np.hypot(h / 2, w / 2) + 1e-6)
        b_hits = int(r.bbox[0] <= 1) + int(r.bbox[1] <= 1) + int(r.bbox[2] >= h - 2) + int(r.bbox[3] >= w - 2)
        if b_hits >= 2:
            continue  # ชิ้นส่วนที่แตะขอบภาพตั้งแต่ 2 ด้านขึ้นไป (ริมฝีปาก/ขอบเสื้อ) ตัดทิ้งทันที
        penalty = 0.05 if b_hits == 1 else 1.0
        score = float(r.area) * (max(0.0, 1.0 - dist) ** 1.2) * penalty
        candidates.append((b_hits, dist, score, r.label))
    
    if not candidates:
        return None

    # ลำดับความสำคัญ: ถ้ามีชิ้นส่วนตรงกลางที่ไม่แตะขอบภาพเลย (เช่น ไฝแท้บนผิวหนัง)
    # ให้เลือกชิ้นส่วนที่ไม่แตะขอบเสมอ เพื่อป้องกันไม่ให้ริมฝีปากหรือขอบเสื้อที่อยู่ขอบภาพมาแย่งคะแนน
    non_border = [c for c in candidates if c[0] == 0]
    if non_border:
        best_cand = max(non_border, key=lambda c: c[2])
    else:
        best_cand = max(candidates, key=lambda c: c[2])
    
    best = best_cand[3]
    mask = (lab == best).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, cnts, -1, 255, -1)  # เติมรูภายใน
    return mask

def segment_lesion(img_rgb, iters=2, sigma=40):
    """แยกรอยโรคแบบชดเชยแสงไม่สม่ำเสมอ (flat-field correction) ตาม Colab ข้อ 2 เป๊ะ
    1) ประมาณแสงพื้นหลังด้วย Gaussian blur ขนาดใหญ่ แล้วหาส่วนที่มืดกว่าพื้นหลัง
    2) ประมาณพื้นหลังใหม่จากผิวหนังเท่านั้น (ตัดรอยโรคออก) แล้วแยกซ้ำ 2 รอบ
    """
    if not OPENCV_AVAILABLE:
        h, w = img_rgb.shape[:2]
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
        return ((dist <= min(h, w) // 4).astype(np.uint8) * 255)

    h, w = img_rgb.shape[:2]
    L = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    bg = cv2.GaussianBlur(L, (0, 0), sigma)
    mask = _otsu_pick(bg - L, h, w)
    for _ in range(iters):
        if mask is None:
            break
        ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        skin = (cv2.dilate(mask, ring) == 0).astype(np.float32)
        if skin.mean() < 0.05:  # รอยโรคเต็มภาพ ใช้ผลรอบก่อน
            break
        bg = (cv2.GaussianBlur(L * skin, (0, 0), sigma) /
              (cv2.GaussianBlur(skin, (0, 0), sigma) + 1e-6))
        new = _otsu_pick(bg - L, h, w)
        if new is None:
            break
        mask = new

    if mask is None:
        return None

    return mask

# =============================================================
# 3) A – Asymmetry ตาม Colab
# =============================================================
def asymmetry_features(mask):
    m = (mask > 0).astype(np.uint8)
    props_list = regionprops(m)
    if not props_list:
        return {'A_asym_horizontal': 0.1, 'A_asym_vertical': 0.1, 'A_asym_mean': 0.1}
    props = props_list[0]
    cy, cx = props.centroid
    angle = np.degrees(props.orientation)
    # หมุนให้แกนหลักของรอยโรคตั้งตรง แล้ววางจุดศูนย์กลางไว้กลางภาพ
    h, w = m.shape
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    r = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    area = r.sum() + 1e-6
    asym_h = float(np.logical_xor(r, np.fliplr(r)).sum() / area)
    asym_v = float(np.logical_xor(r, np.flipud(r)).sum() / area)
    return {
        'A_asym_horizontal': asym_h,
        'A_asym_vertical': asym_v,
        'A_asym_mean': (asym_h + asym_v) / 2
    }

# =============================================================
# 4) B – Border + D – Diameter ตาม Colab
# =============================================================
def border_diameter_features(mask):
    m = (mask > 0).astype(np.uint8)
    props_list = regionprops(m)
    if not props_list:
        return {
            'B_compactness': 1.0, 'B_solidity': 0.95, 'B_convexity': 0.95,
            'B_eccentricity': 0.2, 'B_extent': 0.8, 'D_area': 1000.0,
            'D_perimeter': 100.0, 'D_equiv_diameter': 35.0,
            'D_major_axis': 40.0, 'D_minor_axis': 35.0,
        }
    p = props_list[0]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {
            'B_compactness': 1.0, 'B_solidity': 0.95, 'B_convexity': 0.95,
            'B_eccentricity': 0.2, 'B_extent': 0.8, 'D_area': 1000.0,
            'D_perimeter': 100.0, 'D_equiv_diameter': 35.0,
            'D_major_axis': 40.0, 'D_minor_axis': 35.0,
        }
    cnt = max(cnts, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    area = p.area
    hull = cv2.convexHull(cnt)
    hull_perim = cv2.arcLength(hull, True)
    return {
        'B_compactness': float(perimeter ** 2 / (4 * np.pi * area + 1e-6)), # วงกลม = 1
        'B_solidity': float(p.solidity), # area / convex area
        'B_convexity': float(hull_perim / (perimeter + 1e-6)),
        'B_eccentricity': float(p.eccentricity),
        'B_extent': float(p.extent), # area / bbox area
        'D_area': float(area),
        'D_perimeter': float(perimeter),
        'D_equiv_diameter': float(getattr(p, 'equivalent_diameter_area', getattr(p, 'equivalent_diameter', 35.0))),
        'D_major_axis': float(p.axis_major_length if hasattr(p, 'axis_major_length') else p.major_axis_length),
        'D_minor_axis': float(p.axis_minor_length if hasattr(p, 'axis_minor_length') else p.minor_axis_length),
    }

# =============================================================
# 5) C – Color (เฉพาะพิกเซลในรอยโรค) ตาม Colab
# =============================================================
def color_features(img_rgb, mask):
    inside = mask > 0
    feats = {}
    spaces = {
        'RGB': (img_rgb, ['R', 'G', 'B']),
        'HSV': (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV), ['H', 'S', 'V']),
        'Lab': (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB), ['L', 'a', 'b']),
    }
    for _, (img, names) in spaces.items():
        for i, ch in enumerate(names):
            v = img[..., i][inside].astype(np.float32)
            if len(v) > 0:
                feats[f'C_{ch}_mean'] = float(v.mean())
                feats[f'C_{ch}_std'] = float(v.std())
                feats[f'C_{ch}_skew'] = float(skew(v)) if v.std() > 0 else 0.0
            else:
                feats[f'C_{ch}_mean'] = 0.0
                feats[f'C_{ch}_std'] = 0.0
                feats[f'C_{ch}_skew'] = 0.0

    # ความแตกต่างของสีระหว่างรอยโรคกับผิวรอบ ๆ
    outside = ~inside
    lab = spaces['Lab'][0].astype(np.float32)
    if outside.sum() > 0 and inside.sum() > 0:
        feats['C_contrast_to_skin'] = float(np.linalg.norm(
            lab[inside].mean(0) - lab[outside].mean(0)))
    else:
        feats['C_contrast_to_skin'] = 0.0

    # จำนวนสีที่พบ: quantize ใน Lab แล้วนับสีที่มีสัดส่วน > 5%
    if inside.sum() > 0:
        q = (lab[inside] // 32).astype(int)
        _, counts = np.unique(q, axis=0, return_counts=True)
        feats['C_n_colors'] = int((counts / counts.sum() > 0.05).sum())
    else:
        feats['C_n_colors'] = 1

    return feats

# =============================================================
# 6) Texture – GLCM + LBP ตาม Colab
# =============================================================
def texture_features(img_rgb, mask):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        crop = gray
    else:
        crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    crop_q = (crop // 4).astype(np.uint8) # ลดเป็น 64 ระดับเทา
    glcm = graycomatrix(
        crop_q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    feats = {}
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        feats[f'T_glcm_{prop}'] = float(graycoprops(glcm, prop).mean())

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    if mask.sum() > 0:
        hist, _ = np.histogram(lbp[mask > 0], bins=np.arange(11), density=True)
    else:
        hist = np.zeros(10)

    for i, v in enumerate(hist):
        feats[f'T_lbp_{i}'] = float(v)

    return feats

# =============================================================
# 7) รวมทุก feature ของภาพ 1 ภาพ (รวม 58 features) ตาม Colab
# =============================================================
def extract_handcrafted(img_rgb):
    # ลบขน
    cleaned = remove_hair(img_rgb)
    # ตัดขอบรอยโรค
    mask = segment_lesion(cleaned)

    feats = {}
    feats.update(asymmetry_features(mask))
    feats.update(border_diameter_features(mask))
    feats.update(color_features(cleaned, mask))
    feats.update(texture_features(cleaned, mask))
    return feats, mask, cleaned

# ตรวจสอบว่าเป็นภาพผิวหนังหรือไม่ (Skin Color Threshold - HSV + YCrCb + RGB)
def check_is_skin(img_rgb):
    if not OPENCV_AVAILABLE:
        return True
    
    # 1. ตรวจสอบใน HSV Space
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([0, 20, 50], dtype=np.uint8)
    upper_hsv = np.array([28, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 2. ตรวจสอบใน YCrCb Space (มาตรฐานสากลด้านการตรวจจับผิวหนังมนุษย์)
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # 3. ตรวจสอบเงื่อนไขพิกเซลสีผิว (RGB Bounds)
    r = img_rgb[:, :, 0].astype(np.int16)
    g = img_rgb[:, :, 1].astype(np.int16)
    b = img_rgb[:, :, 2].astype(np.int16)
    rgb_rule = (r > 60) & (g > 35) & (b > 25) & (r > g) & (r > b) & ((r - g) >= 8)

    # รวม Mask ของผิวหนัง
    skin_mask = ((mask_hsv > 0) | (mask_ycrcb > 0)) & rgb_rule
    ratio = float(skin_mask.sum()) / float(img_rgb.shape[0] * img_rgb.shape[1])
    return ratio >= 0.20 # ต้องมีพิกเซลผิวหนังอย่างน้อย 20%

# ตรวจสอบการมีอยู่ของเม็ดสีเมลานิน (Melanin Pigment & Clinical Scope Guard)
# ไฝและมะเร็งผิวหนังตามเกณฑ์ ISIC (Melanocytic Lesions) ต้องมีเม็ดสีเมลานินเข้ม (น้ำตาล/ดำ/เทา)
# และมีความมืดกว่าผิวรอบข้างชัดเจน ในขณะที่ผื่นแพ้ สิว หรือการอักเสบ (Erythematous/Inflammatory) 
# จะมีสีแดง/ชมพูนำเด่น และไม่มีเม็ดสีเข้มของไฝ
def check_melanin_pigmentation(img_rgb, mask):
    if not OPENCV_AVAILABLE:
        return True, "valid"
    inside = mask > 0
    outside = ~inside
    if inside.sum() < 25 or outside.sum() < 50:
        return True, "valid"

    # คำนวณในระบบสี Lab
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L_in = lab[..., 0][inside]
    L_out = lab[..., 0][outside]
    a_in = lab[..., 1][inside]
    a_out = lab[..., 1][outside]

    mean_L_in = float(L_in.mean())
    mean_L_out = float(L_out.mean())
    mean_a_in = float(a_in.mean())
    mean_a_out = float(a_out.mean())

    lum_drop = mean_L_out - mean_L_in  # ความมืดเมื่อเทียบกับผิวรอบข้าง
    redness_excess = mean_a_in - mean_a_out  # ความแดงส่วนเกินใน Lab a*

    # คำนวณความแดงเทียบกับเขียวและน้ำเงิน (Erythema Index ใน RGB)
    r_in = img_rgb[..., 0][inside].astype(np.float32)
    g_in = img_rgb[..., 1][inside].astype(np.float32)
    b_in = img_rgb[..., 2][inside].astype(np.float32)
    erythema = (r_in - g_in) + (r_in - b_in)
    mean_erythema = float(erythema.mean())

    # เม็ดสีเมลานินเข้ม (Dark Melanin Pigment): พิกเซลที่มีค่า L* < 115 หรือเฉดสีคล้ำเข้ม
    dark_pigment_pixels = ((L_in < 115) | ((r_in < 130) & (g_in < 100) & (b_in < 90))).sum()
    dark_pigment_ratio = float(dark_pigment_pixels) / float(inside.sum())

    # 1. กรณีเป็นผื่นแดงกว้าง หรือสิวอักเสบ (Pure Erythema / Inflammatory Condition):
    # แดงชัดเจน แต่แทบไม่มีเม็ดสีเข้มของไฝเลย และไม่มืดกว่าผิวรอบข้าง
    is_pure_erythema = (
        (mean_erythema > 35.0 or redness_excess > 5.0) and
        (dark_pigment_ratio < 0.06) and
        (lum_drop < 8.0)
    )

    # 2. กรณีตุ่มแดงสว่าง (Bright Red Papule / Acne): สว่างเท่าหรือมากกว่าผิวรอบข้าง แต่แดงจัด
    is_bright_red = (
        (lum_drop <= 2.0) and
        (mean_erythema > 40.0 or redness_excess > 6.0) and
        (dark_pigment_ratio < 0.08)
    )

    # 3. กรณีผื่นแดงกระจายตัวที่ไม่มีเม็ดสีเมลานิน (Diffuse Non-pigmented Rash)
    is_diffuse_rash = (
        (dark_pigment_ratio < 0.03) and
        (lum_drop < 5.0) and
        (mean_erythema > 28.0 or redness_excess > 4.0)
    )

    if is_pure_erythema or is_bright_red or is_diffuse_rash:
        return False, "erythematous_rash_or_acne"

    return True, "pigmented_lesion"

@app.get("/")
def read_root():
    try:
        model_loaded = (get_trained_pipeline() is not None)
        cnn = get_trained_cnn()
        cnn_loaded = (cnn is not None)
        classes_clean = [int(x) if str(x).isdigit() else str(x) for x in loaded_classes] if loaded_classes else [0, 1]
        thr_clean = float(loaded_threshold) if loaded_threshold is not None else 0.5
        return {
            "status": "online",
            "project": "Skin Cancer: Malignant vs. Benign (ISIC/Kaggle)",
            "model_loaded": model_loaded,
            "cnn_loaded": cnn_loaded,
            "cnn_type": loaded_cnn_type if cnn_loaded else None,
            "ensemble_mode": f"70% CNN + 30% SVM (w_cnn={loaded_w_cnn})" if cnn_loaded else "SVM Only (fallback)",
            "model_file": str(loaded_model_name) if loaded_model_name else None,
            "model_size_bytes": int(loaded_model_size),
            "threshold": float(loaded_t_lo),
            "threshold_screening": float(loaded_t_lo),
            "threshold_precision": float(loaded_t_hi),
            "version": "v4.1-calibrated-screening",
            "has_ood_guard": (loaded_ood is not None),
            "classes": classes_clean,
            "features_count": 58,
            "cnn_status_note": loaded_cnn_error,
            "criteria": "Ensemble v4: Deep Learning (EfficientNetB0) + Handcrafted ABCD/Texture"
        }
    except Exception as e:
        return {
            "status": "online",
            "error_note": str(e),
            "model_loaded": (loaded_pipeline is not None)
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # ปรับขนาดมาตรฐานการประมวลผล
    img_rgb = np.array(image.resize((224, 224)))

    # ตรวจสอบว่าเป็นภาพผิวหนังหรือไม่
    is_skin = check_is_skin(img_rgb)
    if not is_skin:
        return {
            "is_skin": False,
            "predicted_disease": "ไม่สามารถประเมินผลได้",
            "confidence": 0.0,
            "risk_level": "uncertain",
            "message": "ภาพที่ถ่ายไม่อยู่ในขอบเขตของผิวหนัง หรือมีแสงสว่างไม่เพียงพอ กรุณาถ่ายภาพรอยโรคผิวหนังใหม่อีกครั้ง",
            "gradcam_image": None
        }

    # สกัด Features 58 ตัว + รอยโรคที่แบ่งส่วนได้ตามสูตรโครงงาน
    if FEATURES_MODULE_AVAILABLE:
        cleaned = features.remove_hair(img_rgb)
        mask, info = features.segment_lesion(cleaned, return_info=True)
        seg_q = float(info.get('quality', 1.0)) if info else 0.0
        seg_method = str(info.get('method', '')) if info else 'none'
        if mask is None or mask.sum() == 0:
            return {
                "is_skin": True,
                "has_lesion": False,
                "predicted_disease": "ผิวปกติ / ไม่พบรอยโรค",
                "confidence": 0.0,
                "risk_level": "uncertain",
                "message": "ตรวจไม่พบรอยโรคหรือไฝบนผิวหนัง\n\n📌 ระบบไม่สามารถระบุขอบเขตของไฝได้อย่างชัดเจน เพื่อความปลอดภัยทางการแพทย์จึงไม่เดาสุ่มผล กรุณาถ่ายภาพระยะประชิด 10–15 ซม. โดยจัดให้ไฝอยู่ตรงกลางภาพค่ะ",
                "gradcam_image": None
            }
        feats = {}
        feats.update(features.asymmetry_features(mask))
        feats.update(features.border_diameter_features(mask))
        feats.update(features.color_features(cleaned, mask))
        feats.update(features.texture_features(cleaned, mask))
    else:
        cleaned = remove_hair(img_rgb)
        mask = segment_lesion(cleaned)
        if mask is None or mask.sum() == 0:
            return {
                "is_skin": True,
                "has_lesion": False,
                "predicted_disease": "ผิวปกติ / ไม่พบรอยโรค",
                "confidence": 0.0,
                "risk_level": "uncertain",
                "message": "ตรวจไม่พบรอยโรคหรือไฝบนผิวหนัง\n\n📌 ระบบไม่สามารถระบุขอบเขตของไฝได้อย่างชัดเจน เพื่อความปลอดภัยทางการแพทย์จึงไม่เดาสุ่มผล กรุณาถ่ายภาพระยะประชิด 10–15 ซม. โดยจัดให้ไฝอยู่ตรงกลางภาพค่ะ",
                "gradcam_image": None
            }
        feats = {}
        feats.update(asymmetry_features(mask))
        feats.update(border_diameter_features(mask))
        feats.update(color_features(cleaned, mask))
        feats.update(texture_features(cleaned, mask))
        seg_q = 1.0
        seg_method = 'legacy'

    # ตรวจสอบการมีอยู่ของรอยโรค
    inside = mask > 0
    if inside.sum() < 30:
        return {
            "is_skin": True,
            "has_lesion": False,
            "predicted_disease": "ผิวปกติ / ไม่พบรอยโรค",
            "confidence": 0.0,
            "risk_level": "uncertain",
            "message": "ตรวจไม่พบรอยโรคหรือไฝบนผิวหนัง กรุณาถ่ายภาพใหม่โดยจัดตำแหน่งกล้องให้เห็นไฝหรือรอยโรคชัดเจนค่ะ"
        }

    # วาดเส้นขอบสีเขียวรอบรอยโรค (เหมือนรูปในข้อ 2 ของ Colab)
    show = img_rgb.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cv2.drawContours(show, cnts, -1, (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
    overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    # ทำนายผลด้วยโมเดล Pipeline ที่ฝึกไว้จริง (skin_cancer_pipeline.joblib)
    pipeline = get_trained_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="ไม่พบไฟล์โมเดล AI (skin_cancer_pipeline.joblib) บนเซิร์ฟเวอร์ ระบบจำเป็นต้องใช้โมเดลจริงที่ฝึกไว้เท่านั้น"
        )

    try:
        import pandas as pd
        # 1. เลือกฟีเจอร์และเรียงคอลัมน์ให้ตรงกับตอนเทรนโมเดลใน Colab 100%
        active_feat_cols = loaded_feat_cols if (loaded_feat_cols and len(loaded_feat_cols) > 0) else DEFAULT_FEAT_COLS
        df_feat = pd.DataFrame([feats])
        for col in active_feat_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        df_feat = df_feat[active_feat_cols]
        # ล้างค่า NaN และ Infinity ที่อาจเกิดจากการคำนวณทางสถิติ เพื่อไม่ให้โมเดล Error
        df_feat = df_feat.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # ตัวตรวจภาพนอกขอบเขต (OOD): วัดระยะ Mahalanobis ของฟีเจอร์ภาพนี้จากชุดเทรน
        # ใช้งานเมื่อไฟล์โมเดลมีค่า 'ood' ที่สร้างจากเซลล์ Colab (ไม่มีก็ข้ามไป)
        if loaded_ood:
            try:
                x = (df_feat.values[0] - np.asarray(loaded_ood['mean'])) / np.asarray(loaded_ood['scale'])
                diff = x - np.asarray(loaded_ood['center'])
                ood_dist = float(np.sqrt(max(diff @ np.asarray(loaded_ood['inv_cov']) @ diff, 0.0)))
                if ood_dist > float(loaded_ood['threshold']):
                    return {
                        "is_skin": True,
                        "is_out_of_scope": True,
                        "predicted_disease": "ไม่สามารถประเมินผลได้",
                        "confidence": 0.0,
                        "risk_level": "uncertain",
                        "ood_distance": round(ood_dist, 2),
                        "ood_threshold": round(float(loaded_ood['threshold']), 2),
                        "message": "ลักษณะภาพนี้ต่างจากรอยโรคที่โมเดลเคยเรียนมาก (เช่น สิว ผื่น หรือผิวที่ไม่ใช่ไฝ) จึงไม่เดาผล เพื่อไม่ให้ได้คำแนะนำที่ผิด หากกังวลควรปรึกษาแพทย์ผิวหนังโดยตรงค่ะ",
                        "gradcam_image": None
                    }
            except Exception as ood_err:
                print(f"[WARN] OOD check skipped: {ood_err}")

        # 2. ค้นหาตำแหน่ง index ของคลาส malignant ใน classes_ ของโมเดลจริง
        malignant_idx = 1
        for i, c in enumerate(loaded_classes):
            if c == 1 or str(c).lower() in ('1', 'malignant', 'melanoma'):
                malignant_idx = i
                break

        benign_idx = 0 if malignant_idx != 0 else 1

        # 3. ส่งข้อมูลเข้าโมเดล SVM (Handcrafted Features)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df_feat.values)[0]
            prob_hand = float(probs[malignant_idx])
        else:
            pred = pipeline.predict(df_feat.values)[0]
            prob_hand = 1.0 if (pred == loaded_classes[malignant_idx]) else 0.0

        # 4. ส่งข้อมูลเข้าโมเดล CNN (ถ้ามีไฟล์โมเดลและ runtime พร้อม)
        prob_cnn = predict_cnn_prob(img_rgb)

        # 5. รวมผลแบบ Ensemble (ตามสัดส่วน w_cnn ใน bundle.joblib เช่น 70% CNN + 30% SVM)
        if prob_cnn is not None and loaded_w_cnn > 0.0:
            prob_malignant = float(loaded_w_cnn * prob_cnn + (1.0 - loaded_w_cnn) * prob_hand)
            active_model_desc = f"Ensemble v4 ({int(loaded_w_cnn*100)}% CNN + {int((1.0-loaded_w_cnn)*100)}% SVM)"
        else:
            prob_malignant = prob_hand
            active_model_desc = "Handcrafted ABCD + SVM Pipeline (SVM 100%)"

        # เกณฑ์ตัดสินตาม Colab v4 (t_lo = 0.22, t_hi = 0.51):
        # ในการคัดกรองมะเร็งผิวหนัง (Cancer Screening) ต้องใช้ t_lo เพื่อให้ Recall >= 98% (ไม่หลุดมะเร็ง)
        is_malignant = (prob_malignant >= loaded_t_lo)

        # คำนวณค่าความเชื่อมั่นแบบปรับเทียบตามเกณฑ์คัดกรอง (Calibrated Screening Confidence)
        if is_malignant:
            if prob_malignant >= loaded_t_hi:
                confidence = float(prob_malignant * 100.0)
            else:
                # ช่วงตรวจพบความเสี่ยงสูงกว่าเกณฑ์คัดกรอง (t_lo <= P < t_hi)
                calibrated = 50.0 + ((prob_malignant - loaded_t_lo) / max(1.0 - loaded_t_lo, 1e-6)) * 50.0
                confidence = float(min(max(calibrated, 50.0), 99.0))
        else:
            calibrated = 50.0 + ((loaded_t_lo - prob_malignant) / max(loaded_t_lo, 1e-6)) * 50.0
            confidence = float(min(max(calibrated, 50.0), 99.0))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"โมเดลไม่สามารถประมวลผลการจำแนกได้: {str(e)}"
        )

    # ผลลัพธ์การจำแนกตามเกณฑ์โมเดล ISIC (Decision Threshold จาก Colab)
    if seg_q < 0.15 and not is_malignant:
        is_uncertain = True
        disease_name = "ความมั่นใจต่ำกว่าเกณฑ์ความปลอดภัย (Uncertain)"
        risk_level = "uncertain"
        message = "แยกขอบเขตรอยโรคออกจากผิวหนังได้ไม่ชัดเจน เพื่อความปลอดภัยทางการแพทย์จึงไม่ฟันธงผล กรุณาถ่ายภาพระยะประชิดให้ชัดเจนอีกครั้งค่ะ"
    else:
        is_uncertain = False
        disease_name = "มะเร็งผิวหนัง (Malignant Melanoma)" if is_malignant else "ไฝปกติ / ไม่พบเนื้อร้าย (Benign)"
        risk_level = "high" if is_malignant else "low"
        message = (
            "ตรวจพบความผิดปกติที่สอดคล้องกับเนื้อร้ายตามเกณฑ์ ABCD และโครงสร้างเซลล์ (ความเสี่ยงสูงกว่าเกณฑ์ความปลอดภัย) ควรรีบพบแพทย์ผิวหนังโดยเร็ว"
            if is_malignant else
            "รอยโรคมีลักษณะสมมาตร ขอบมนเรียบ สีสม่ำเสมอ ปลอดภัยไม่พบสัญญาณมะเร็งผิวหนัง"
        )

    asym_val = feats.get('A_asym_mean', 0.0)
    comp_val = feats.get('B_compactness', 1.0)
    colors_count = int(feats.get('C_n_colors', 1))
    equiv_d = feats.get('D_equiv_diameter', 0.0)
    glcm_contrast = feats.get('T_glcm_contrast', 0.0)

    return {
        "is_skin": True,
        "is_uncertain": is_uncertain,
        "predicted_disease": disease_name,
        "confidence": round(confidence, 1),
        "risk_level": risk_level,
        "message": message,
        "threshold": round(loaded_threshold, 3),
        "threshold_screening": round(loaded_t_lo, 3),
        "threshold_precision": round(loaded_t_hi, 3),
        "prob_malignant": round(prob_malignant, 4),
        "prob_cnn": round(prob_cnn, 4) if prob_cnn is not None else None,
        "prob_hand": round(prob_hand, 4),
        "model_type": active_model_desc,
        "gradcam_image": overlay_b64,
        "abcd_metrics": {
            "asymmetry": {
                "score": round(float(asym_val), 3),
                "status": "สมมาตรปกติ" if asym_val < 0.25 else "ไม่สมมาตร (เสี่ยง)",
                "desc": f"ค่าดัชนีความไม่สมมาตร {round(float(asym_val), 3)}"
            },
            "border": {
                "score": round(float(comp_val), 2),
                "status": "ขอบเรียบสม่ำเสมอ" if comp_val < 1.4 else "ขอบหยัก/ไม่เรียบ (เสี่ยง)",
                "desc": f"ค่าความกะทัดรัดของขอบ {round(float(comp_val), 2)}"
            },
            "color": {
                "score": colors_count,
                "status": "สีสม่ำเสมอเฉดเดียว" if colors_count <= 1 else f"พบหลายเฉดสี ({colors_count} สี)",
                "desc": f"จำนวนกลุ่มสีหลักที่สกัดได้ {colors_count} เฉด"
            },
            "diameter": {
                "score": round(float(equiv_d), 1),
                "status": "ขนาดปกติ" if equiv_d < 100 else "ขนาดค่อนข้างใหญ่",
                "desc": f"เส้นผ่านศูนย์กลางเทียบเคียง {round(float(equiv_d), 1)} px"
            },
            "texture": {
                "score": round(float(glcm_contrast), 2),
                "status": "พื้นผิวเรียบเนียน" if glcm_contrast < 12.0 else "พื้นผิวขรุขระ/มีความแปรปรวน",
                "desc": f"ค่า GLCM Contrast {round(float(glcm_contrast), 2)}"
            }
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)