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
    # ใช้ MORPH_OPEN ขนาด (5, 5) เพื่อตัดคอคอดหรือเส้นเชื่อมระหว่างรอยโรคกับริมฝีปาก/เงา
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=2)
    # ใช้ MORPH_CLOSE ขนาดเล็ก (3, 3) เพียงรอบเดียวเพื่อเติมรูเล็กๆ ภายในตัวไฝ ไม่ให้เชื่อมข้ามไปยังริมฝีปาก
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    return mask


def _quality(mask):
    """ให้คะแนน mask 0..1: อยู่กลางภาพ, ขนาดพอดี, ไม่แนบขอบภาพ, รูปทรงไม่แตกกระจาย"""
    h, w = mask.shape
    m = mask > 0
    area = float(m.sum())
    frac = area / float(h * w)
    # อนุญาตให้จับไฝขนาดเล็กตั้งแต่ 0.2% ของภาพ (ประมาณ 100 พิกเซล) ถึง 65%
    if frac < 0.002 or frac > 0.65:
        return 0.0
    ys, xs = np.where(m)
    if len(ys) == 0:
        return 0.0
    cy, cx = float(ys.mean()), float(xs.mean())
    dist = np.hypot(cy - h / 2, cx - w / 2) / np.hypot(h / 2, w / 2)

    # ตรวจสอบการติดขอบภาพ (ขอบ 3 px)
    edge = np.zeros_like(m)
    edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = True
    touch_px = int((m & edge).sum())
    borders_hit = int(ys.min() <= 2) + int(ys.max() >= h - 3) + int(xs.min() <= 2) + int(xs.max() >= w - 3)

    # หากติดขอบภาพตั้งแต่ 2 ด้านขึ้นไป หรือติดขอบด้านใดด้านหนึ่งเกิน 15 พิกเซล (เช่น ริมฝีปากที่ขอบล่าง) ให้คะแนนเป็น 0 ทันที
    if borders_hit >= 2 or touch_px > 15:
        return 0.0
    
    if touch_px > 0:
        border_penalty = 0.05
    else:
        border_penalty = 1.0

    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) + 1e-6
    solidity = min(1.0, area / hull_area)

    # ให้คะแนนตามขนาด: รองรับไฝทั่วไป (0.5% - 25% ของภาพ) ได้คะแนนเต็ม 1.0
    if 0.005 <= frac <= 0.25:
        size_score = 1.0
    elif frac < 0.005:
        size_score = max(0.3, frac / 0.005)
    else:
        size_score = max(0.1, 1.0 - (frac - 0.25) / 0.40)

    return float(size_score * (max(0.0, 1.0 - dist) ** 1.2) * border_penalty * solidity)


def _pick(M, valid, h, w):
    """Otsu แบบ Multi-Level เพื่อแยกเม็ดไฝสีเข้มออกจากเนื้อเยื่อข้างเคียง (เช่น ริมฝีปาก)"""
    M_smooth = cv2.GaussianBlur(M.astype(np.float32), (5, 5), 0)
    vals = M_smooth[valid]
    if vals.size < 50 or vals.std() < 1e-3:
        return None
    t0 = threshold_otsu(vals)
    val_max = float(vals.max())
    
    # ทดสอบ 3 ระดับ Threshold เพื่อค้นหาเม็ดสีที่เข้มที่สุด (ไฝ) แยกขาดจากริมฝีปาก
    thresholds = [t0]
    if val_max > t0:
        thresholds.append(t0 + 0.25 * (val_max - t0))
        thresholds.append(t0 + 0.45 * (val_max - t0))

    best, best_s = None, 0.0
    for t in thresholds:
        raw_mask = ((M_smooth > t) & valid).astype(np.uint8) * 255
        cleaned_mask = _clean(raw_mask)
        lab = label(cleaned_mask > 0)
        for r in regionprops(lab):
            if r.area < 0.002 * h * w:
                continue
            comp = (lab == r.label).astype(np.uint8) * 255
            s = _quality(comp)
            if t > t0:
                s *= 1.2  # โบนัสความเข้มคมชัดของเม็ดสีไฝ
            if s > best_s:
                best, best_s = comp, s

    if best is None:
        return None
    cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
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
    if cnts:
        cv2.drawContours(out, cnts, -1, 255, -1)
    return out


MIN_QUALITY = 0.08


def segment_lesion(img_rgb, return_info=False):
    """แยกรอยโรคผิวหนังอย่างแม่นยำระดับการแพทย์ (Dermatological Melanin Detection)
    1. วัดค่าความมืดของเม็ดสีเมลานินเทียบกับผิวหนังปกติรอบข้าง
    2. คัดแยกเฉพาะก้อนเม็ดสีเข้ม (ไฝ) โดยไม่เชื่อมต่อกับริมฝีปาก เลือด หรือขอบภาพ
    3. ปฏิเสธวัตถุที่แตะขอบภาพอย่างเด็ดขาด
    """
    h, w = img_rgb.shape[:2]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]

    # ตัดขอบมืดของกล้องออก
    valid = ~_frame_mask(L)
    if valid.sum() < 100:
        valid = np.ones((h, w), bool)

    # 1. ประมาณค่าความสว่างเฉลี่ยของผิวหนังปกติ (Skin Baseline)
    skin_L = float(np.median(L[valid]))

    # 2. แผนที่ความมืดเทียบกับผิว (Melanin Darkness Map)
    darkness = np.clip(skin_L - L, 0.0, 255.0)
    darkness_valid = darkness[valid]
    max_dark = float(darkness_valid.max()) if darkness_valid.size > 0 else 0.0

    # หากภาพนี้ไม่มีส่วนใดมืดกว่าผิวอย่างมีนัยสำคัญ (น้อยกว่า 12 หน่วย)
    if max_dark < 12.0:
        return (None, {'method': 'none', 'quality': 0.0}) if return_info else None

    # 3. ค้นหา Component ของไฝโดยเน้นระดับความเข้มของเม็ดสีเมลานิน (Melanin Focus)
    # ไฝจะมีความเข้มสูงกว่าริมฝีปากหรือเงามาก เริ่มจากระดับเข้มสุดลงมา
    best_comp = None
    best_score = 0.0
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for ratio in [0.60, 0.45, 0.32, 0.20]:
        thresh = max(14.0, max_dark * ratio)
        raw_mask = ((darkness >= thresh) & valid).astype(np.uint8) * 255
        
        # ทำความสะอาดด้วย Opening ขนาดเล็ก 3x3 เพียง 1 รอบ เพื่อลบ noise โดยไม่กลืนไฝเล็ก
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_clean, iterations=1)
        
        lab_map = label(cleaned > 0)
        for r in regionprops(lab_map):
            area = float(r.area)
            frac = area / float(h * w)
            # ขนาดไฝต้องอยู่ในช่วง 0.1% ถึง 50% ของภาพ
            if frac < 0.001 or frac > 0.50:
                continue

            # ตรวจสอบการแตะขอบภาพ (ริมฝีปาก, ขอบเสื้อ, ขอบภาพ)
            touches_border = (r.bbox[0] <= 1 or r.bbox[1] <= 1 or 
                              r.bbox[2] >= h - 2 or r.bbox[3] >= w - 2)
            if touches_border:
                continue  # วัตถุที่แตะขอบภาพถูกตัดทิ้งทันที!

            cy, cx = r.centroid
            dist = np.hypot(cy - h / 2, cx - w / 2) / np.hypot(h / 2, w / 2)

            comp = (lab_map == r.label).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) + 1e-6
            solidity = min(1.0, area / hull_area)

            # ให้คะแนน: ยิ่งทึบแน่นกลม (solidity สูง), เม็ดสีเข้มคมชัด (ratio สูง), ไม่ติดขอบ ยิ่งได้คะแนนสูง
            score = float(solidity * (max(0.0, 1.0 - 0.4 * dist)) * (1.0 + ratio))
            if score > best_score:
                best_score = score
                best_comp = comp

        # หากพบลักษณะไฝที่ชัดเจนและทึบแน่นแล้ว ให้เลือกทันที ไม่ต้องลด threshold ให้ไปโดนปาก
        if best_comp is not None and best_score > 0.70:
            break

    if best_comp is None:
        # หากไม่พบรอยโรคที่ชัดเจน คืนค่า None ทันที ห้ามสร้างวงรีปลอมเด็ดขาด
        return (None, {'method': 'none', 'quality': 0.0}) if return_info else None

    # เติมรูภายในตัวไฝ
    cnts, _ = cv2.findContours(best_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cv2.drawContours(best_comp, cnts, -1, 255, -1)

    return (best_comp, {'method': 'melanin_otsu', 'quality': float(best_score)}) if return_info else best_comp


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
