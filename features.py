# =============================================================
#  Handcrafted Features (ABCD + Texture) — ใช้ไฟล์เดียวกันทั้งใน Colab และเว็บแอป
# =============================================================
import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from joblib import Parallel, delayed

DATA_DIR = '/content/data'
CLASSES = ['benign', 'malignant']          # 0 = benign, 1 = malignant
IMG_SIZE = 224
META_COLS = ('file', 'label', 'seg_method', 'seg_quality')


# 1) Preprocessing: ลบขน (DullRazor แบบย่อ)
def remove_hair(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_rgb, hair_mask, 3, cv2.INPAINT_TELEA)


# 2) Segmentation


def _clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def _quality(mask):
    """ให้คะแนน mask 0..1: อยู่กลางภาพ, ขนาดพอดี, ไม่แนบขอบภาพ, รูปทรงไม่แตกกระจาย"""
    h, w = mask.shape
    m = mask > 0
    area = m.sum()
    frac = area / (h * w)
    if frac < 0.015 or frac > 0.70:
        return 0.0
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    dist = np.hypot(cy - h / 2, cx - w / 2) / np.hypot(h / 2, w / 2)
    # สัดส่วนของ mask ที่ติดขอบภาพ (ขอบ 2 px)
    edge = np.zeros_like(m)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    border_touch = (m & edge).sum() / max(1, 2 * (h + w) * 2)
    border_touch = min(1.0, border_touch * 8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) + 1e-6
    solidity = area / hull_area
    size_score = 1.0 - abs(frac - 0.18) / 0.55          # ชอบ ~18% ของภาพ
    return float(max(0.0, size_score) * (1 - dist) ** 1.5 *
                 (1 - border_touch) * solidity)


def _pick(M, valid, h, w):
    """Otsu เฉพาะ pixel ที่ valid แล้วเลือก component ที่อยู่กลางภาพและไม่แนบขอบ"""
    M = cv2.GaussianBlur(M.astype(np.float32), (5, 5), 0)
    vals = M[valid]
    if vals.size < 100 or vals.std() < 1e-3:
        return None
    t = threshold_otsu(vals)
    mask = ((M > t) & valid).astype(np.uint8) * 255
    mask = _clean(mask)
    lab = label(mask > 0)
    best, best_s = None, 0.0
    for r in regionprops(lab):
        if r.area < 0.01 * h * w:
            continue
        comp = (lab == r.label).astype(np.uint8) * 255
        s = _quality(comp)
        if s > best_s:
            best, best_s = comp, s
    if best is None:
        return None
    cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(best, cnts, -1, 255, -1)       # เติมรู
    return best


def _frame_mask(L):
    """pixel ที่เป็นขอบมืด (vignette / ขอบดำของกล้อง) -> invalid"""
    h, w = L.shape
    med = np.median(L)
    dark = (L < 0.45 * med).astype(np.uint8)
    lab = label(dark)
    frame = np.zeros((h, w), bool)
    for r in regionprops(lab):
        b = r.bbox
        if (b[0] == 0 or b[1] == 0 or b[2] == h or b[3] == w) and r.area > 30:
            frame |= (lab == r.label)
    return cv2.dilate(frame.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def _flat_map(L, valid, sigma=30):
    """ชดเชยแสงไม่สม่ำเสมอ: ประมาณพื้นหลังจาก pixel ที่ valid เท่านั้น"""
    v = valid.astype(np.float32)
    bg = cv2.GaussianBlur(L * v, (0, 0), sigma) / (cv2.GaussianBlur(v, (0, 0), sigma) + 1e-6)
    return bg - L


def _grabcut(img_rgb):
    h, w = img_rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[:h // 8, :] = mask[-h // 8:, :] = cv2.GC_BGD
    mask[:, :w // 8] = mask[:, -w // 8:] = cv2.GC_BGD
    cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 4), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 10, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), mask, None,
                    bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    m = _clean(m)
    lab = label(m > 0)
    if lab.max() == 0:
        return None
    best = max(regionprops(lab), key=lambda r: r.area)
    out = (lab == best.label).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


MIN_QUALITY = 0.12


def segment_lesion(img_rgb, return_info=False):
    h, w = img_rgb.shape[:2]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    S = hsv[..., 1]
    valid = ~_frame_mask(L)
    if valid.mean() < 0.3:
        valid = np.ones((h, w), bool)

    maps = {
        'L': _flat_map(L, valid),                         # เข้มกว่าผิวรอบข้าง
        'a': -_flat_map(A, valid),                        # แดงกว่า
        'S': -_flat_map(S, valid),                        # อิ่มสีกว่า
        'Lab': np.sqrt(sum(_flat_map(c, valid) ** 2 for c in (L, A, B))),  # ต่างสีรวม
    }
    cands = []
    for name, M in maps.items():
        m = _pick(M, valid, h, w)
        if m is not None:
            cands.append((_quality(m), name, m))
    method = 'none'
    mask, q = None, 0.0
    if cands:
        q, method, mask = max(cands, key=lambda t: t[0])
    if mask is None or q < 0.02:
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 5, h // 5), 0, 0, 360, 255, -1)
        method, q = 'fallback_ellipse', 0.05
    return (mask, {'method': method, 'quality': q}) if return_info else mask


# 3) A – Asymmetry
def asymmetry_features(mask):
    m = (mask > 0).astype(np.uint8)
    props = regionprops(m)[0]
    cy, cx = props.centroid
    angle = np.degrees(props.orientation)
    h, w = m.shape
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    r = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    area = r.sum() + 1e-6
    asym_h = np.logical_xor(r, np.fliplr(r)).sum() / area
    asym_v = np.logical_xor(r, np.flipud(r)).sum() / area
    return {'A_asym_horizontal': asym_h,
            'A_asym_vertical': asym_v,
            'A_asym_mean': (asym_h + asym_v) / 2}


# 4) B – Border + D – Diameter
def border_diameter_features(mask):
    m = (mask > 0).astype(np.uint8)
    p = regionprops(m)[0]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    area = p.area
    hull = cv2.convexHull(cnt)
    hull_perim = cv2.arcLength(hull, True)
    return {
        'B_compactness': perimeter ** 2 / (4 * np.pi * area + 1e-6),
        'B_solidity': p.solidity,
        'B_convexity': hull_perim / (perimeter + 1e-6),
        'B_eccentricity': p.eccentricity,
        'B_extent': p.extent,
        'D_area': area,
        'D_perimeter': perimeter,
        'D_equiv_diameter': p.equivalent_diameter_area,
        'D_major_axis': p.axis_major_length,
        'D_minor_axis': p.axis_minor_length,
    }


# 5) C – Color (เฉพาะพิกเซลในรอยโรค)
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
            feats[f'C_{ch}_mean'] = v.mean()
            feats[f'C_{ch}_std'] = v.std()
            feats[f'C_{ch}_skew'] = skew(v) if v.std() > 0 else 0.0
    outside = ~inside
    lab = spaces['Lab'][0].astype(np.float32)
    feats['C_contrast_to_skin'] = (np.linalg.norm(lab[inside].mean(0) - lab[outside].mean(0))
                                   if outside.sum() > 0 else 0.0)
    q = (lab[inside] // 32).astype(int)
    _, counts = np.unique(q, axis=0, return_counts=True)
    feats['C_n_colors'] = int((counts / counts.sum() > 0.05).sum())
    return feats


# 6) Texture – GLCM + LBP
def texture_features(img_rgb, mask):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(mask > 0)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_q = (crop // 4).astype(np.uint8)
    glcm = graycomatrix(crop_q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    feats = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        feats[f'T_glcm_{prop}'] = graycoprops(glcm, prop).mean()
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp[mask > 0], bins=np.arange(11), density=True)
    for i, v in enumerate(hist):
        feats[f'T_lbp_{i}'] = v
    return feats


# 7) รวมทุก feature ของภาพ 1 ภาพ (รับ path หรือ array RGB)
def extract_with_meta(src):
    if isinstance(src, str):
        img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(src)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = remove_hair(img)
    mask, info = segment_lesion(img, return_info=True)
    feats = {}
    feats.update(asymmetry_features(mask))
    feats.update(border_diameter_features(mask))
    feats.update(color_features(img, mask))
    feats.update(texture_features(img, mask))
    feats['seg_method'] = info['method']
    feats['seg_quality'] = round(info['quality'], 4)
    return feats


def extract_handcrafted(src):
    f = extract_with_meta(src)
    f.pop('seg_method'); f.pop('seg_quality')
    return f


def build_feature_table(split):
    paths = []
    for y, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, split, cls)
        for f in sorted(os.listdir(folder)):
            paths.append((os.path.join(folder, f), y))
    results = Parallel(n_jobs=-1)(delayed(extract_with_meta)(p) for p, _ in paths)
    df = pd.DataFrame(results)
    df.insert(0, 'label', [y for _, y in paths])
    df.insert(0, 'file', [os.path.basename(p) for p, _ in paths])
    return df
