import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
import os

# =============================
# COLAB SETUP
# =============================
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')


# =============================================================================
# SEGMENTASI
# =============================================================================
def get_egg_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1]

    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return mask

    c = max(cnts, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [c], -1, 255, -1)

    return final_mask


# =============================================================================
# EKSTRAKSI FITUR
# =============================================================================
def extract_features(img_rgb, mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    S = hsv[:,:,1][mask > 0]
    V = hsv[:,:,2][mask > 0]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.bitwise_and(blackhat, blackhat, mask=mask)

    _, spot_mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    spot_mask = cv2.medianBlur(spot_mask, 3)

    cnts, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spot_area = 0
    valid = []

    for c in cnts:
        area = cv2.contourArea(c)
        if 10 < area < 500:
            valid.append(c)
            spot_area += area

    egg_area = np.sum(mask > 0)

    return {
        "S_mean": float(np.mean(S)),
        "V_mean": float(np.mean(V)),
        "spot_area_ratio": spot_area / (egg_area + 1e-6),
        "bintik_count": len(valid),
        "avg_spot_size": spot_area / (len(valid)+1e-6)
    }, spot_mask


# =============================================================================
# CLASSIFICATION + EXPLANATION
# =============================================================================
def classify(f):
    S, V = f['S_mean'], f['V_mean']
    ratio, count = f['spot_area_ratio'], f['bintik_count']

    if S < 120 and V > 150:
        return "Stres", "Warna pucat (S rendah)"

    if ratio > 0.02:
        return "Cacingan", "Area bintik besar"

    if count >= 6 and ratio > 0.005:
        return "Cacingan", "Bintik banyak"

    if count >= 8:
        return "Cacingan", "Bintik sangat banyak"

    return "Sehat", "Warna normal & bintik sedikit"


# =============================================================================
# VISUAL INFORMATIVE
# =============================================================================
def process_image(path):
    img = np.array(Image.open(path).convert('RGB').resize((300,300)))

    mask = get_egg_mask(img)
    f, spot_mask = extract_features(img, mask)
    pred, reason = classify(f)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    fig = plt.figure(figsize=(14,6))

    # BARIS 1
    plt.subplot(2,4,1); plt.imshow(img); plt.title("Original"); plt.axis('off')
    plt.subplot(2,4,2); plt.imshow(mask, cmap='gray'); plt.title("Mask"); plt.axis('off')
    plt.subplot(2,4,3); plt.imshow(hsv[:,:,1], cmap='YlOrBr'); plt.title(f"S={f['S_mean']:.1f}"); plt.axis('off')
    plt.subplot(2,4,4); plt.imshow(spot_mask, cmap='hot'); plt.title(f"Count={f['bintik_count']}"); plt.axis('off')

    # BARIS 2 (TEXT)
    plt.subplot(2,1,2)
    plt.axis('off')

    plt.text(0,0.9,
             f"HASIL ANALISIS\n"
             f"----------------------\n"
             f"Diagnosis : {pred}\n"
             f"Alasan    : {reason}\n\n"
             f"S (warna) : {f['S_mean']:.1f}\n"
             f"Count     : {f['bintik_count']}\n"
             f"Ratio     : {f['spot_area_ratio']:.4f}",
             fontsize=12)

    display(fig)
    plt.close()

    return pred


# =============================================================================
# TESTING (ANTI ERROR)
# =============================================================================
def run_testing(base_path):
    # 🔥 mapping folder → label model
    folder_map = {
        "Sehat": "Sehat",
        "Setres": "Stres",   # 🔥 FIX TYPO
        "Cacingan": "Cacingan"
    }

    total = 0
    correct = 0

    print("\nTESTING")
    print("Base path:", base_path)

    if not os.path.exists(base_path):
        print("❌ BASE PATH SALAH!")
        return

    for folder_name, true_label in folder_map.items():
        folder = os.path.join(base_path, folder_name)

        if not os.path.exists(folder):
            print(f"❌ Folder tidak ada: {folder}")
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            if not os.path.isfile(path):
                continue

            img = np.array(Image.open(path).convert('RGB').resize((300,300)))
            mask = get_egg_mask(img)
            f, _ = extract_features(img, mask)
            pred, _ = classify(f)

            total += 1

            # 🔥 COMPARISON YANG BENAR
            if pred.strip().lower() == true_label.strip().lower():
                correct += 1

            print(f"{file} | GT:{true_label} | Pred:{pred}")

    if total == 0:
        print("\n❌ Tidak ada data ditemukan!")
    else:
        print(f"\nAccuracy: {correct}/{total} = {(correct/total)*100:.2f}%")

# =============================================================================
# MAIN
# =============================================================================
base_path = "/content/drive/MyDrive/Citra_egg"  # 🔥 FIX PATH

if __name__ == "__main__":
    run_testing(base_path)

    from google.colab import files
    uploaded = files.upload()

    for f in uploaded.keys():
        print("\n===== VISUAL ANALISIS =====")
        process_image(f)