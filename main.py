import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# 1. SEGMENTASI (FINAL: EDGE BASED + SAFE FALLBACK)
# =============================================================================
def get_egg_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # =========================
    # EDGE DETECTION
    # =========================
    edges = cv2.Canny(gray, 30, 100)

    # perbesar edge
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8))

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # =========================
    # FALLBACK: kalau gagal
    # =========================
    if not cnts:
        h, w = gray.shape

        # pakai area tengah (AMAN, bukan ngawur)
        mask = np.zeros_like(gray)
        cv2.circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        return mask, c

    # ambil contour terbesar
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    return mask, c


# =============================================================================
# 2. EKSTRAKSI FITUR
# =============================================================================
def extract_features(img_rgb, mask, contour):
    if contour is None:
        return {}

    features = {}

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # warna
    features['S_mean'] = hsv[:,:,1][mask>0].mean()

    # kontras
    features['contrast'] = np.std(gray[mask>0])

    # =========================
    # DETEKSI BINTIK
    # =========================
    v = hsv[:,:,2]
    s = hsv[:,:,1]

    bintik_mask = (v < 140) & (s > 40) & (mask > 0)
    bintik_mask = bintik_mask.astype(np.uint8) * 255

    bintik_mask = cv2.morphologyEx(bintik_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    cnts, _ = cv2.findContours(bintik_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spots = [c for c in cnts if 20 < cv2.contourArea(c) < 500]
    features['bintik_count'] = len(spots)

    # tekstur
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features['roughness'] = np.var(lap[mask>0]) / 100.0

    return features


# =============================================================================
# 3. CLASSIFICATION (FINAL FIXED)
# =============================================================================
def classify(features):
    if not features:
        return "Tidak Terdeteksi", {}

    S = features['S_mean']
    contrast = features['contrast']
    spots = features['bintik_count']
    rough = features['roughness']

    # CACINGAN
    if spots > 5:
        return "Cacingan", features

    # TUA (FIX)
    if rough > 0.8 and contrast < 50:
        return "Tua", features

    # KALSIUM
    if S < 90 and contrast < 40:
        return "Kalsium", features

    # STRES
    if S < 110:
        return "Stres", features

    return "Sehat", features


# =============================================================================
# 4. VISUALISASI
# =============================================================================
def process_image(path):
    img = np.array(Image.open(path).convert('RGB').resize((300,300)))

    mask, contour = get_egg_mask(img)
    feats = extract_features(img, mask, contour)
    diagnosis, _ = classify(feats)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # =========================
    # BINTIK VISUAL
    # =========================
    bintik_mask = ((v < 140) & (s > 40) & (mask > 0)).astype(np.uint8) * 255

    # =========================
    # EDGE
    # =========================
    edges = cv2.Canny(gray, 30, 100)

    # =========================
    # TEKSTUR (LAPLACIAN)
    # =========================
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vis = cv2.convertScaleAbs(lap)

    # =========================
    # PLOT
    # =========================
    fig = plt.figure(figsize=(18,10))
    gs = gridspec.GridSpec(2,4)

    ax1 = plt.subplot(gs[0,0])
    ax1.imshow(img)
    ax1.set_title("Original")
    ax1.axis('off')

    ax2 = plt.subplot(gs[0,1])
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("Mask")
    ax2.axis('off')

    ax3 = plt.subplot(gs[0,2])
    ax3.imshow(s, cmap='YlOrBr')
    ax3.set_title(f"Saturation\nMean: {feats.get('S_mean',0):.1f}")
    ax3.axis('off')

    ax4 = plt.subplot(gs[0,3])
    ax4.imshow(v, cmap='gray')
    ax4.set_title("Brightness (V)")
    ax4.axis('off')

    ax5 = plt.subplot(gs[1,0])
    ax5.imshow(bintik_mask, cmap='hot')
    ax5.set_title(f"Bintik\nCount: {feats.get('bintik_count',0)}")
    ax5.axis('off')

    ax6 = plt.subplot(gs[1,1])
    ax6.imshow(edges, cmap='gray')
    ax6.set_title("Edge Detection")
    ax6.axis('off')

    ax7 = plt.subplot(gs[1,2])
    ax7.imshow(lap_vis, cmap='inferno')
    ax7.set_title(f"Texture (Roughness)\n{feats.get('roughness',0):.2f}")
    ax7.axis('off')

    ax8 = plt.subplot(gs[1,3])
    ax8.axis('off')

    text = f"""
DIAGNOSIS: {diagnosis}

FEATURES:
S_mean     : {feats.get('S_mean',0):.1f}
Contrast   : {feats.get('contrast',0):.1f}
Spots      : {feats.get('bintik_count',0)}
Roughness  : {feats.get('roughness',0):.2f}
"""

    ax8.text(0.05,0.9,text,fontsize=12,va='top',
             bbox=dict(boxstyle='round', facecolor='#e8f6f3'))

    plt.tight_layout()
    return fig

# =============================================================================
# 5. AUTO TESTING
# =============================================================================
def run_evaluation():
    print("\n" + "="*60)
    print("EVALUASI DATASET")
    print("="*60)

    test_files = {
        "sehat.png": "Sehat",
        "kalsium.png": "Kalsium",
        "cacingan.png": "Cacingan",
        "stres.png": "Stres",
        "tua.png": "Tua"
    }

    correct = 0
    results = []

    for file, label in test_files.items():
        try:
            img = np.array(Image.open(file).convert('RGB').resize((300,300)))

            mask, contour = get_egg_mask(img)
            feats = extract_features(img, mask, contour)
            pred, _ = classify(feats)

            ok = (pred == label)
            if ok:
                correct += 1

            results.append((file, label, pred, ok))

        except:
            results.append((file, label, "ERROR", False))

    print("\nHASIL:")
    print("-"*60)

    for r in results:
        print(f"{r[0]:15} | GT:{r[1]:10} | Pred:{r[2]:10} | {'✓' if r[3] else '✗'}")

    total = len(test_files)
    print("-"*60)
    print(f"AKURASI: {correct}/{total} = {(correct/total)*100:.2f}%")

    print("="*60)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    run_evaluation()

    from google.colab import files
    uploaded = files.upload()

    for f in uploaded.keys():
        fig = process_image(f)
        plt.show()