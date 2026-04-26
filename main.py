# =============================================================================
#  IDENTIFIKASI KONDISI KESEHATAN UNGGAS AYAM
#  Berdasarkan Karakteristik Telur — Pendekatan Pengolahan Citra Digital
# =============================================================================
#  Program Studi Teknologi Informasi
#  Universitas Bina Sarana Informatika — Kelompok 3
#
#  ALUR PIPELINE (Pengolahan Citra First):
#  ┌─────────────────────────────────────────────────────────────────┐
#  │ STEP 1 : Akuisisi Citra        → tampilkan gambar asli          │
#  │ STEP 2 : Pre-processing        → Grayscale, Blur, CLAHE         │
#  │ STEP 3 : Segmentasi            → Thresholding, Morfologi,       │
#  │                                   Kontur, masking               │
#  │ STEP 4 : Ekstraksi Fitur       → Warna, GLCM, Bentuk            │
#  │ STEP 5 : Klasifikasi (ML)      → Training & Diagnosis akhir     │
#  └─────────────────────────────────────────────────────────────────┘
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — INSTALASI (jalankan sekali di Google Colab)
# ─────────────────────────────────────────────────────────────────────────────
# !pip install opencv-python-headless scikit-image scikit-learn
# !pip install matplotlib numpy pandas seaborn Pillow

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — IMPORT LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns

print("=" * 65)
print("  IDENTIFIKASI KESEHATAN UNGGAS — PENGOLAHAN CITRA STEP-BY-STEP")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — DEFINISI KELAS DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────

KELAS = {
    0: {
        'nama'        : 'Sehat / Normal',
        'warna_ui'    : '#27ae60',
        'ikon'        : '✅',
        'tanda_telur' : 'Cangkang coklat keemasan, tebal, permukaan halus merata, bentuk elips simetris, bersih.',
        'penyebab'    : 'Ayam dalam kondisi prima: nutrisi seimbang, suhu kandang ideal, bebas penyakit.',
        'rekomendasi' : [
            'Pertahankan kualitas pakan (Ca 3.5–4.5%, Vit D3 cukup).',
            'Jaga suhu kandang 18–24°C dan ventilasi baik.',
            'Lanjutkan program vaksinasi dan biosekuriti rutin.',
        ],
        'bahaya' : 'Rendah',
    },
    1: {
        'nama'        : 'Kekurangan Kalsium (Hypocalcemia)',
        'warna_ui'    : '#e67e22',
        'ikon'        : '🦴',
        'tanda_telur' : 'Cangkang sangat tipis, kasar/bergelombang, kadang shell-less egg.',
        'penyebab'    : 'Asupan kalsium dalam pakan < 3%, kualitas batu kapur buruk.',
        'rekomendasi' : [
            'Naikkan kadar kalsium pakan ke 3.8–4.2%.',
            'Berikan suplemen kalsium cair di air minum pagi hari.',
            'Periksa pH pakan dan air minum (ideal 6.5–7.0).',
        ],
        'bahaya' : 'Sedang',
    },
    2: {
        'nama'        : 'Kekurangan Vitamin D3',
        'warna_ui'    : '#f39c12',
        'ikon'        : '☀️',
        'tanda_telur' : 'Cangkang tipis tapi lebih pucat/keputihan, kadang telur lunak di ujungnya.',
        'penyebab'    : 'Vitamin D3 dalam pakan tidak mencukupi (<500 IU/kg).',
        'rekomendasi' : [
            'Tambahkan Vitamin D3 dalam pakan: minimal 1.500–2.000 IU/kg.',
            'Berikan akses sinar matahari pagi minimal 2 jam/hari.',
            'Injeksi Vit D3 + Vit A via air minum selama 5 hari.',
        ],
        'bahaya' : 'Sedang',
    },
    3: {
        'nama'        : 'Stres Panas (Heat Stress)',
        'warna_ui'    : '#e74c3c',
        'ikon'        : '🌡️',
        'tanda_telur' : 'Cangkang sangat pucat, tipis, telur lebih kecil, produksi menurun drastis.',
        'penyebab'    : 'Suhu kandang >30°C → alkalosis respiratori → berkurangnya HCO₃⁻.',
        'rekomendasi' : [
            'Pasang cooling system; targetkan suhu <28°C.',
            'Berikan air minum dingin (15–18°C) sepanjang hari.',
            'Suplementasikan NaHCO₃ 0.1–0.3% di air minum.',
        ],
        'bahaya' : 'Tinggi',
    },
    4: {
        'nama'        : 'Infeksi Penyakit (Newcastle / IB)',
        'warna_ui'    : '#8e44ad',
        'ikon'        : '🦠',
        'tanda_telur' : 'Bentuk sangat cacat, cangkang kasar seperti amplas, warna belang-belang.',
        'penyebab'    : 'Infeksi IB/ND merusak oviduct, mengganggu proses pembentukan telur.',
        'rekomendasi' : [
            '⚠️ SEGERA karantina dan hubungi dokter hewan!',
            'Lakukan vaksinasi ND (La Sota/B1) dan IB (H120) sesuai jadwal.',
            'Biosekuriti ketat: disinfeksi kandang, batasi akses masuk.',
        ],
        'bahaya' : 'SANGAT TINGGI',
    },
    5: {
        'nama'        : 'Kebersihan Buruk / Kontaminasi',
        'warna_ui'    : '#7f8c8d',
        'ikon'        : '🧫',
        'tanda_telur' : 'Cangkang bernoda coklat/hitam/kehijauan, berbau, permukaan lengket.',
        'penyebab'    : 'Kandang kotor, litter basah, telur tidak segera dipungut.',
        'rekomendasi' : [
            'Pungut telur minimal 3–4× sehari.',
            'Bersihkan & disinfeksi kandang 2× seminggu.',
            'Ganti litter yang lembap segera; kelembapan ideal <25%.',
        ],
        'bahaya' : 'Tinggi — Risiko Food Safety',
    },
}

CLASS_NAMES = [KELAS[k]['nama'] for k in KELAS]
N_CLASSES   = len(CLASS_NAMES)

print(f"\n  Total kelas diagnosis: {N_CLASSES}")
for k, v in KELAS.items():
    print(f"  Kelas {k}: {v['ikon']} {v['nama']}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — GENERATOR GAMBAR SINTETIS (DATA MENTAH / AKUISISI CITRA)
# ─────────────────────────────────────────────────────────────────────────────

def buat_telur(size=(300, 300), warna=(200, 180, 150), brightness=1.0,
               shape_ratio=1.0, add_crack=False, add_stain=False,
               roughness=0, pale_tip=False, spotty_shell=False):
    """Buat gambar telur sintetis sebagai pengganti foto asli."""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 235
    cx, cy = size[1] // 2, size[0] // 2

    ax = int(80 * min(shape_ratio, 1.5))
    ay = int(100 * max(0.6, 1.0 / max(shape_ratio, 0.01)))
    w  = tuple(int(c * brightness) for c in warna)

    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, w, -1)

    for i in range(25, 0, -1):
        a  = i / 25.0
        hc = tuple(int(c + (255 - c) * a * 0.25) for c in w)
        cv2.ellipse(img, (cx - 18, cy - 22),
                    (int(35 * a), int(45 * a)), 0, 0, 360, hc, -1)

    if roughness > 0:
        for _ in range(roughness * 20):
            ox = np.random.randint(-ax + 5, ax - 5)
            oy = np.random.randint(-ay + 5, ay - 5)
            if (ox / ax) ** 2 + (oy / ay) ** 2 < 0.85:
                r_bump  = np.random.randint(2, roughness * 3 + 3)
                bump_c  = tuple(max(0, c - np.random.randint(15, 35)) for c in w)
                cv2.circle(img, (cx + ox, cy + oy), r_bump, bump_c, -1)

    if pale_tip:
        for i in range(20):
            alpha = i / 20.0
            pale  = tuple(int(c + (245 - c) * alpha) for c in w)
            cv2.ellipse(img, (cx, cy - ay + i * 4), (int(ax * 0.5), 5),
                        0, 0, 360, pale, -1)

    if spotty_shell:
        for _ in range(np.random.randint(8, 18)):
            ox = np.random.randint(-ax + 10, ax - 10)
            oy = np.random.randint(-ay + 10, ay - 10)
            if (ox / ax) ** 2 + (oy / ay) ** 2 < 0.75:
                sc = tuple(np.random.randint(80, 140) for _ in range(3))
                cv2.ellipse(img, (cx + ox, cy + oy),
                            (np.random.randint(5, 18), np.random.randint(3, 10)),
                            np.random.randint(0, 180), 0, 360, sc, -1)

    if add_crack:
        for _ in range(np.random.randint(1, 4)):
            pts = [(cx + np.random.randint(-30, 30),
                    cy + np.random.randint(-50, 50))]
            for __ in range(np.random.randint(3, 6)):
                pts.append((pts[-1][0] + np.random.randint(-15, 15),
                             pts[-1][1] + np.random.randint(-20, 20)))
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (60, 40, 25), 2)

    if add_stain:
        for _ in range(np.random.randint(4, 10)):
            ox = np.random.randint(-ax + 5, ax - 5)
            oy = np.random.randint(-ay + 5, ay - 5)
            if (ox / ax) ** 2 + (oy / ay) ** 2 < 0.75:
                choice = np.random.choice([0, 1, 2])
                sc = [(40, 25, 10), (20, 70, 20), (120, 20, 20)][choice]
                cv2.circle(img, (cx + ox, cy + oy), np.random.randint(10, 28), sc, -1)
    return img


def buat_dataset(n_per_class=35):
    """Buat dataset sintetis — pengganti akuisisi foto asli."""
    images, labels = [], []
    rng = np.random.default_rng(42)
    specs = {
        0: dict(warna_range=((185, 215), (165, 195), (135, 165)),
                brightness=(0.90, 1.00), shape_ratio=(0.92, 1.08),
                crack=0.00, stain=0.00, roughness=0, pale_tip=False, spotty=False),
        1: dict(warna_range=((175, 205), (155, 185), (125, 155)),
                brightness=(0.70, 0.85), shape_ratio=(0.85, 1.15),
                crack=0.60, stain=0.10, roughness=4, pale_tip=False, spotty=False),
        2: dict(warna_range=((220, 248), (215, 242), (210, 238)),
                brightness=(0.55, 0.72), shape_ratio=(0.90, 1.10),
                crack=0.20, stain=0.00, roughness=2, pale_tip=True,  spotty=False),
        3: dict(warna_range=((230, 252), (225, 248), (218, 242)),
                brightness=(0.50, 0.65), shape_ratio=(0.75, 0.90),
                crack=0.15, stain=0.00, roughness=3, pale_tip=False, spotty=False),
        4: dict(warna_range=((140, 185), (120, 165), (95, 140)),
                brightness=(0.65, 0.82), shape_ratio=(0.50, 0.70),
                crack=0.70, stain=0.30, roughness=5, pale_tip=False, spotty=True),
        5: dict(warna_range=((155, 195), (135, 175), (105, 145)),
                brightness=(0.60, 0.78), shape_ratio=(0.88, 1.12),
                crack=0.10, stain=1.00, roughness=1, pale_tip=False, spotty=False),
    }
    for label, sp in specs.items():
        for _ in range(n_per_class):
            r   = rng.integers(*sp['warna_range'][0])
            g   = rng.integers(*sp['warna_range'][1])
            b   = rng.integers(*sp['warna_range'][2])
            br  = rng.uniform(*sp['brightness'])
            sr  = rng.uniform(*sp['shape_ratio'])
            img = buat_telur(
                warna=(int(r), int(g), int(b)),
                brightness=br, shape_ratio=sr,
                add_crack=rng.random() < sp['crack'],
                add_stain=rng.random() < sp['stain'],
                roughness=sp['roughness'],
                pale_tip=sp['pale_tip'],
                spotty_shell=sp['spotty'],
            )
            images.append(img)
            labels.append(label)
    return images, labels


def muat_dataset_asli(folder_dataset="dataset"):
    images_asli = []
    labels_asli = []
    
    print(f"\n[CELL 4] Membaca dataset FOTO ASLI dari folder '{folder_dataset}'...")
    for k in range(N_CLASSES):
        folder_kelas = os.path.join(folder_dataset, str(k))
        if not os.path.exists(folder_kelas):
            print(f"  [!] Folder {folder_kelas} tidak ditemukan. Lewati.")
            continue
            
        # Cari semua format foto jpg/jpeg/png
        file_paths = glob.glob(os.path.join(folder_kelas, "*.[jp][pn]*")) + \
                     glob.glob(os.path.join(folder_kelas, "*.[jJ][pP][gG]"))
                     
        for path in file_paths:
            # Buka, jadikan RGB, dan resize ukuran ke 300x300 agar konsisten
            img = np.array(Image.open(path).convert('RGB').resize((300, 300)))
            images_asli.append(img)
            labels_asli.append(k)
            
    return images_asli, labels_asli

# Panggil fungsi untuk membaca foto asli
images, labels = muat_dataset_asli("dataset")
print(f"  Total: {len(images)} gambar asli berhasil dimuat | {N_CLASSES} kelas")


# ═══════════════════════════════════════════════════════════════════════════════
#   ██████████████████  STEP 1 : AKUISISI CITRA  ████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — TAMPILKAN CITRA ASLI (HASIL AKUISISI) — 1 SAMPEL PER KELAS

print("\n" + "═" * 65)
print("  STEP 1 — AKUISISI CITRA")
print("  Menampilkan 1 citra asli (original) untuk setiap kelas...")
print("═" * 65)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("STEP 1 — AKUISISI CITRA\nCitra Asli (Original) — 6 Kondisi Kesehatan Ayam",
             fontsize=13, fontweight='bold')

for ax, k in zip(axes.flat, range(N_CLASSES)):
    idx  = labels.index(k)
    info = KELAS[k]
    ax.imshow(images[idx])
    ax.set_title(f"{info['ikon']}  Kelas {k}: {info['nama']}\n"
                 f"Bahaya: {info['bahaya']}",
                 fontsize=9, color=info['warna_ui'], fontweight='bold')
    ax.axis('off')
    # Border warna sesuai kelas
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(info['warna_ui'])
        spine.set_linewidth(4)

plt.tight_layout()
plt.savefig('STEP1_akuisisi_citra.png', dpi=150, bbox_inches='tight')
plt.show()
print("[✓] STEP 1 selesai — gambar disimpan: STEP1_akuisisi_citra.png")


# ═══════════════════════════════════════════════════════════════════════════════
#   ████████████████████  STEP 2 : PRE-PROCESSING  ██████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — FUNGSI PREPROCESSING

def preprocess(img_rgb):
    """
    Tahap Pre-processing:
    1. Konversi ke Grayscale  → menyederhanakan data warna jadi intensitas
    2. Gaussian Blur          → mengurangi noise (derau) pada citra
    3. CLAHE                  → meningkatkan kontras secara lokal (adaptif)
    4. Otsu Thresholding      → menghasilkan citra biner (hitam-putih)
    5. Morfologi (Close+Open) → membersihkan noise dan menutup lubang kecil

    Mengembalikan: (grayscale, enhanced_clahe, binary_mask, blurred)
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Gaussian Blur (noise reduction)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. CLAHE — Contrast Limited Adaptive Histogram Equalization
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 4. Otsu Thresholding → binary mask
    _, binary = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morfologi: Close → menutup lubang kecil | Open → hapus noise kecil
    kernel  = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    return gray, blurred, enhanced, binary, cleaned


def visualisasi_preprocessing(img_rgb, judul_kelas="", simpan=None):
    """
    Tampilkan step-by-step pre-processing dalam satu figure:
    Kolom: Original | Grayscale | Gaussian Blur | CLAHE | Otsu Binary | Final Mask
    """
    gray, blurred, enhanced, binary, cleaned = preprocess(img_rgb)

    # -- Menyimpan setiap output langkah sebagai file gambar terpisah --
    if simpan:
        base_name = simpan.replace('.png', '')
        cv2.imwrite(f"{base_name}_0_original.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{base_name}_1_grayscale.png", gray)
        cv2.imwrite(f"{base_name}_2_blur.png", blurred)
        cv2.imwrite(f"{base_name}_3_clahe.png", enhanced)
        cv2.imwrite(f"{base_name}_4_otsu.png", binary)
        cv2.imwrite(f"{base_name}_5_morfologi.png", cleaned)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle(f"STEP 2 — PRE-PROCESSING  |  {judul_kelas}",
                 fontsize=12, fontweight='bold')

    steps = [
        (img_rgb,   'Cmap',    'ORIGINAL\n(Akuisisi Citra)'),
        (gray,      'gray',    '① GRAYSCALE\n(RGB → Intensitas)'),
        (blurred,   'gray',    '② GAUSSIAN BLUR\n(Noise Reduction)'),
        (enhanced,  'gray',    '③ CLAHE\n(Contrast Enhancement)'),
        (binary,    'gray',    '④ OTSU THRESHOLDING\n(Segmentasi Awal)'),
        (cleaned,   'gray',    '⑤ MORFOLOGI\n(Close + Open)'),
    ]

    for ax, (img, cmap, title) in zip(axes, steps):
        if cmap == 'Cmap':
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=8, fontweight='bold', pad=6)
        ax.axis('off')
        # Garis pembatas antar step
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#3498db')
        ax.spines['bottom'].set_linewidth(2)

    # Panah antar step
    for i in range(len(axes) - 1):
        x_left  = axes[i].get_position().x1
        x_right = axes[i + 1].get_position().x0
        x_mid   = (x_left + x_right) / 2
        fig.text(x_mid, 0.5, '→', ha='center', va='center',
                 fontsize=16, color='#2c3e50', fontweight='bold')

    plt.tight_layout()
    if simpan:
        plt.savefig(simpan, dpi=150, bbox_inches='tight')
    plt.show()


# ── Demo preprocessing untuk SATU sampel tiap kelas ──
print("\n" + "═" * 65)
print("  STEP 2 — PRE-PROCESSING")
print("  Menampilkan pipeline preprocessing per kelas...")
print("═" * 65)

for k in range(N_CLASSES):
    idx  = labels.index(k)
    info = KELAS[k]
    judul = f"Kelas {k}: {info['ikon']} {info['nama']}"
    print(f"  → {judul}")
    visualisasi_preprocessing(images[idx], judul_kelas=judul,
                               simpan=f"STEP2_preprocessing_kelas{k}.png")

print("[✓] STEP 2 selesai — gambar disimpan: STEP2_preprocessing_kelas*.png")


# ═══════════════════════════════════════════════════════════════════════════════
#   █████████████████████  STEP 3 : SEGMENTASI  █████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — FUNGSI SEGMENTASI

def segmentasi(img_rgb, cleaned_mask):
    """
    Segmentasi: memisahkan objek telur dari latar belakang.
    1. Temukan kontur terbesar (= telur)
    2. Buat mask telur (egg_mask)
    3. Overlay kontur pada gambar asli
    4. Terapkan mask → citra telur tersegmentasi

    Mengembalikan: (segmented_img, egg_mask, contour, contour_overlay)
    """
    cnts = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    
    if not contours:
        return img_rgb, cleaned_mask, None, img_rgb.copy()

    # Ambil kontur terbesar
    c = max(contours, key=cv2.contourArea)

    # Buat mask dari kontur telur
    egg_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(egg_mask, [c], -1, 255, -1)

    # Overlay kontur (garis hijau) pada gambar asli
    overlay = img_rgb.copy()
    cv2.drawContours(overlay, [c], -1, (0, 220, 0), 3)  # hijau tebal

    # Gambar bounding box (kotak merah) untuk region telur
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (220, 0, 0), 2)  # merah

    # Hasil akhir: citra tersegmentasi (bg = hitam)
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=egg_mask)

    return segmented, egg_mask, c, overlay


def visualisasi_segmentasi(img_rgb, judul_kelas="", simpan=None):
    """
    Tampilkan step-by-step segmentasi dalam satu figure:
    Original | Cleaned Binary | Kontur Overlay | Egg Mask | Hasil Segmentasi
    """
    gray, blurred, enhanced, binary, cleaned = preprocess(img_rgb)
    segmented, egg_mask, contour, overlay = segmentasi(img_rgb, cleaned)

    # -- Menyimpan setiap output langkah segmentasi secara terpisah --
    if simpan:
        base_name = simpan.replace('.png', '')
        cv2.imwrite(f"{base_name}_1_mask_awal.png", cleaned)
        cv2.imwrite(f"{base_name}_2_overlay_kontur.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{base_name}_3_egg_mask.png", egg_mask)
        cv2.imwrite(f"{base_name}_4_hasil_segmentasi.png", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(f"STEP 3 — SEGMENTASI  |  {judul_kelas}",
                 fontsize=12, fontweight='bold')

    items = [
        (img_rgb,   'Cmap', 'ORIGINAL'),
        (cleaned,   'gray', '① BINARY MASK\n(Hasil Morfologi)'),
        (overlay,   'Cmap', '② DETEKSI KONTUR\n(Garis Hijau = Telur\nKotak Merah = BBox)'),
        (egg_mask,  'gray', '③ EGG MASK\n(Area Telur Saja)'),
        (segmented, 'Cmap', '④ HASIL SEGMENTASI\n(Telur Terisolasi)'),
    ]

    for ax, (img, cmap, title) in zip(axes, items):
        if cmap == 'Cmap':
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=8, fontweight='bold', pad=6)
        ax.axis('off')

    plt.tight_layout()
    if simpan:
        plt.savefig(simpan, dpi=150, bbox_inches='tight')
    plt.show()


print("\n" + "═" * 65)
print("  STEP 3 — SEGMENTASI")
print("  Mendeteksi & mengisolasi objek telur dari background...")
print("═" * 65)

for k in range(N_CLASSES):
    idx   = labels.index(k)
    info  = KELAS[k]
    judul = f"Kelas {k}: {info['ikon']} {info['nama']}"
    print(f"  → {judul}")
    visualisasi_segmentasi(images[idx], judul_kelas=judul,
                            simpan=f"STEP3_segmentasi_kelas{k}.png")

print("[✓] STEP 3 selesai — gambar disimpan: STEP3_segmentasi_kelas*.png")


# ═══════════════════════════════════════════════════════════════════════════════
#   ████████████████████  STEP 4 : EKSTRAKSI FITUR  █████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — FUNGSI EKSTRAKSI FITUR (WARNA, TEKSTUR GLCM, BENTUK GEOMETRI)

FEATURE_NAMES = [
    # Warna RGB (6)
    'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
    # Warna HSV (6)
    'H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std',
    # Warna LAB (3)
    'L_mean', 'a_mean', 'b_mean',
    # Tekstur GLCM (6)
    'GLCM_energy', 'GLCM_contrast', 'GLCM_homogeneity',
    'GLCM_correlation', 'GLCM_dissimilarity', 'GLCM_ASM',
    # Statistik Gray-level (3)
    'gray_mean', 'gray_std', 'gray_entropy',
    # Bentuk Geometri (6)
    'aspect_ratio', 'circularity', 'solidity',
    'extent', 'equiv_diameter', 'perimeter_ratio',
]


def fitur_warna(img_rgb, mask=None):
    """Ekstrak fitur warna: rata-rata & standar deviasi channel RGB, HSV, LAB."""
    px = img_rgb[mask > 0] if mask is not None else img_rgb.reshape(-1, 3)

    rgb = [px[:, i].mean() for i in range(3)] + \
          [px[:, i].std()  for i in range(3)]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hx  = hsv[mask > 0] if mask is not None else hsv.reshape(-1, 3)
    hsv_feats = [hx[:, i].mean() for i in range(3)] + \
                [hx[:, i].std()  for i in range(3)]

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lx  = lab[mask > 0] if mask is not None else lab.reshape(-1, 3)
    lab_feats = [lx[:, i].mean() for i in range(3)]

    return rgb + hsv_feats + lab_feats


def fitur_tekstur_glcm(gray_img, mask=None):
    """
    Ekstrak fitur tekstur menggunakan GLCM (Gray Level Co-occurrence Matrix):
    - Energy, Contrast, Homogeneity, Correlation, Dissimilarity, ASM
    - Statistik intensitas: mean, std, entropy
    """
    if mask is not None:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return [0] * 9
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        roi = gray_img[r0:r1+1, c0:c1+1]
    else:
        roi = gray_img

    roi_s = cv2.resize(roi, (64, 64))

    glcm = graycomatrix(roi_s, [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    g_energy  = graycoprops(glcm, 'energy').mean()
    g_cont    = graycoprops(glcm, 'contrast').mean()
    g_homo    = graycoprops(glcm, 'homogeneity').mean()
    g_corr    = graycoprops(glcm, 'correlation').mean()
    g_diss    = graycoprops(glcm, 'dissimilarity').mean()
    g_asm     = graycoprops(glcm, 'ASM').mean()

    flat    = roi_s.flatten().astype(float)
    g_mean  = flat.mean()
    g_std   = flat.std()
    hist, _ = np.histogram(flat, bins=256, range=(0, 255), density=True)
    hist   += 1e-10
    entropy = -np.sum(hist * np.log2(hist))

    return [g_energy, g_cont, g_homo, g_corr, g_diss, g_asm,
            g_mean, g_std, entropy]


def fitur_bentuk(contour):
    """
    Ekstrak fitur bentuk geometris dari kontur telur:
    - Aspect Ratio, Circularity, Solidity, Extent, Equiv. Diameter, Perimeter Ratio
    """
    if contour is None:
        return [0] * 6

    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if len(contour) >= 5:
        _, (ma, Mi), _ = cv2.fitEllipse(contour)
        aspect_ratio   = max(ma, Mi) / (min(ma, Mi) + 1e-6)
    else:
        aspect_ratio = 1.0

    circularity    = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    hull_area      = cv2.contourArea(cv2.convexHull(contour))
    solidity       = area / (hull_area + 1e-6)
    _, _, w, h     = cv2.boundingRect(contour)
    extent         = area / (w * h + 1e-6)
    equiv_diameter = np.sqrt(4 * area / np.pi)
    perimeter_ratio= perimeter / (equiv_diameter + 1e-6)

    return [aspect_ratio, circularity, solidity,
            extent, equiv_diameter, perimeter_ratio]


def ekstrak_semua_fitur(img_rgb):
    """Pipeline lengkap: preprocessing → segmentasi → ekstraksi semua fitur."""
    gray, blurred, enhanced, binary, cleaned = preprocess(img_rgb)
    segmented, egg_mask, contour, overlay    = segmentasi(img_rgb, cleaned)
    return (fitur_warna(img_rgb, egg_mask) +
            fitur_tekstur_glcm(gray, egg_mask) +
            fitur_bentuk(contour))


def visualisasi_ekstraksi_fitur(img_rgb, judul_kelas="", simpan=None):
    """
    Tampilkan visualisasi ekstraksi fitur untuk satu citra:
    - Histogram warna (RGB)
    - GLCM (co-occurrence matrix sebagian)
    - Fitur bentuk (nilai numerik)
    - Bar chart nilai fitur utama
    """
    gray, blurred, enhanced, binary, cleaned = preprocess(img_rgb)
    segmented, egg_mask, contour, overlay    = segmentasi(img_rgb, cleaned)

    feats_warna   = fitur_warna(img_rgb, egg_mask)
    feats_tekstur = fitur_tekstur_glcm(gray, egg_mask)
    feats_bentuk  = fitur_bentuk(contour)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"STEP 4 — EKSTRAKSI FITUR  |  {judul_kelas}",
                 fontsize=12, fontweight='bold')
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.4)

    # ── [0,0] Gambar Original ──
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_rgb)
    ax_orig.set_title('Citra Input\n(Setelah Segmentasi)', fontsize=8, fontweight='bold')
    ax_orig.axis('off')

    # ── [0,1] Gambar Tersegmentasi ──
    ax_seg = fig.add_subplot(gs[0, 1])
    ax_seg.imshow(segmented)
    ax_seg.set_title('Hasil Segmentasi\n(Area Telur)', fontsize=8, fontweight='bold')
    ax_seg.axis('off')

    # ── [0,2] Histogram Warna RGB ──
    ax_hist = fig.add_subplot(gs[0, 2])
    if egg_mask is not None:
        for ch, col, lbl in zip([0, 1, 2], ['red', 'green', 'blue'], ['R', 'G', 'B']):
            vals = img_rgb[:, :, ch][egg_mask > 0]
            ax_hist.hist(vals, bins=40, color=col, alpha=0.55, label=lbl)
    ax_hist.set_title('① FITUR WARNA\nHistogram RGB', fontsize=8, fontweight='bold')
    ax_hist.set_xlabel('Intensitas (0–255)', fontsize=7)
    ax_hist.set_ylabel('Frekuensi', fontsize=7)
    ax_hist.legend(fontsize=6)
    ax_hist.tick_params(labelsize=6)

    # ── [0,3] Histogram HSV (S & V channel) ──
    ax_hsv = fig.add_subplot(gs[0, 3])
    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    if egg_mask is not None:
        for ch, col, lbl in zip([1, 2], ['orange', 'purple'], ['Saturation', 'Value']):
            vals = hsv_img[:, :, ch][egg_mask > 0]
            ax_hsv.hist(vals, bins=40, color=col, alpha=0.6, label=lbl)
    ax_hsv.set_title('① FITUR WARNA\nHistogram HSV (S & V)', fontsize=8, fontweight='bold')
    ax_hsv.set_xlabel('Nilai', fontsize=7)
    ax_hsv.legend(fontsize=6)
    ax_hsv.tick_params(labelsize=6)

    # ── [0,4] GLCM Tekstur (nilai fitur) ──
    ax_glcm = fig.add_subplot(gs[0, 4])
    glcm_labels = ['Energy', 'Contrast', 'Homogeneity', 'Correlation', 'Dissimilarity']
    glcm_vals   = feats_tekstur[:5]
    # Normalisasi untuk display (0–1)
    max_abs = max(abs(v) for v in glcm_vals) + 1e-6
    glcm_norm = [v / max_abs for v in glcm_vals]
    colors_glcm = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    bars = ax_glcm.barh(glcm_labels, glcm_norm, color=colors_glcm, edgecolor='black', linewidth=0.5)
    ax_glcm.set_title('② FITUR TEKSTUR\nGLCM (Normalized)', fontsize=8, fontweight='bold')
    ax_glcm.set_xlabel('Nilai (ternormalisasi)', fontsize=7)
    for bar, val in zip(bars, glcm_vals):
        ax_glcm.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center', fontsize=6)
    ax_glcm.tick_params(labelsize=6)
    ax_glcm.set_xlim(0, 1.3)

    # ── [1,0-1] Citra Grayscale & GLCM Matrix Visualisasi ──
    ax_gray = fig.add_subplot(gs[1, 0])
    ax_gray.imshow(gray, cmap='gray')
    ax_gray.set_title('Grayscale\n(Input GLCM)', fontsize=8, fontweight='bold')
    ax_gray.axis('off')

    ax_glcm_mat = fig.add_subplot(gs[1, 1])
    gray_rs = cv2.resize(gray, (64, 64))
    
    # Skala nilai piksel ke 0-31 untuk levels=32 agar tidak error (ValueError: image contains values >= levels)
    gray_rs_32 = (gray_rs / 8).astype(np.uint8)
    glcm_mat = graycomatrix(gray_rs_32, [1], [0], levels=32, symmetric=True, normed=True)
    ax_glcm_mat.imshow(np.log1p(glcm_mat[:, :, 0, 0]), cmap='hot', aspect='auto')
    ax_glcm_mat.set_title('② GLCM Matrix\n(log scale, 32 levels)', fontsize=8, fontweight='bold')
    ax_glcm_mat.set_xlabel('Intensitas j', fontsize=7)
    ax_glcm_mat.set_ylabel('Intensitas i', fontsize=7)
    ax_glcm_mat.tick_params(labelsize=6)

    # ── [1,2] Fitur Bentuk (numerik) ──
    ax_shape = fig.add_subplot(gs[1, 2])
    shape_labels = ['Aspect\nRatio', 'Circularity', 'Solidity',
                    'Extent', 'Equiv\nDiam', 'Perim\nRatio']
    shape_vals = feats_bentuk
    colors_shape = ['#1abc9c', '#e74c3c', '#3498db', '#e67e22', '#9b59b6', '#7f8c8d']
    bars2 = ax_shape.bar(shape_labels, shape_vals, color=colors_shape,
                         edgecolor='black', linewidth=0.5)
    ax_shape.set_title('③ FITUR BENTUK\nGeometri Kontur', fontsize=8, fontweight='bold')
    ax_shape.set_ylabel('Nilai', fontsize=7)
    for bar, val in zip(bars2, shape_vals):
        ax_shape.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{val:.2f}', ha='center', fontsize=6)
    ax_shape.tick_params(labelsize=6)

    # ── [1,3-4] Ringkasan semua fitur (tabel mini) ──
    ax_tbl = fig.add_subplot(gs[1, 3:])
    ax_tbl.axis('off')

    tbl_data = [
        ["WARNA (RGB)", f"R={feats_warna[0]:.1f}, G={feats_warna[1]:.1f}, B={feats_warna[2]:.1f}",
         f"σR={feats_warna[3]:.1f}, σG={feats_warna[4]:.1f}, σB={feats_warna[5]:.1f}"],
        ["WARNA (HSV)", f"H={feats_warna[6]:.1f}, S={feats_warna[7]:.1f}, V={feats_warna[8]:.1f}",
         f"σH={feats_warna[9]:.1f}, σS={feats_warna[10]:.1f}, σV={feats_warna[11]:.1f}"],
        ["WARNA (LAB)", f"L={feats_warna[12]:.1f}, a={feats_warna[13]:.1f}, b={feats_warna[14]:.1f}", "—"],
        ["GLCM Energy",    f"{feats_tekstur[0]:.5f}", "Keseragaman tekstur"],
        ["GLCM Contrast",  f"{feats_tekstur[1]:.4f}", "Perbedaan intensitas piksel"],
        ["GLCM Homogen.",  f"{feats_tekstur[2]:.5f}", "Kerataan distribusi"],
        ["GLCM Corr.",     f"{feats_tekstur[3]:.5f}", "Korelasi antar piksel"],
        ["Gray Mean",      f"{feats_tekstur[6]:.2f}",  "Rata-rata intensitas"],
        ["Gray Entropy",   f"{feats_tekstur[8]:.4f}",  "Kompleksitas tekstur"],
        ["Aspect Ratio",   f"{feats_bentuk[0]:.4f}",   "Rasio sumbu panjang/pendek"],
        ["Circularity",    f"{feats_bentuk[1]:.4f}",   "Keidealan bentuk lingkaran"],
        ["Solidity",       f"{feats_bentuk[2]:.4f}",   "Kerapatan area kontur"],
    ]

    tbl = ax_tbl.table(cellText=tbl_data,
                       colLabels=['Fitur', 'Nilai', 'Keterangan'],
                       cellLoc='left', loc='center',
                       bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')
    ax_tbl.set_title('Ringkasan Nilai Fitur yang Diekstrak', fontsize=8, fontweight='bold')

    if simpan:
        plt.savefig(simpan, dpi=150, bbox_inches='tight')
    plt.show()


print("\n" + "═" * 65)
print("  STEP 4 — EKSTRAKSI FITUR (Warna, GLCM, Bentuk)")
print("  Memvisualisasikan fitur yang diekstrak per kelas...")
print("═" * 65)

for k in range(N_CLASSES):
    idx   = labels.index(k)
    info  = KELAS[k]
    judul = f"Kelas {k}: {info['ikon']} {info['nama']}"
    print(f"  → {judul}")
    visualisasi_ekstraksi_fitur(images[idx], judul_kelas=judul,
                                 simpan=f"STEP4_fitur_kelas{k}.png")

print("[✓] STEP 4 selesai — gambar disimpan: STEP4_fitur_kelas*.png")


# ── Ekstraksi Fitur Massal (seluruh dataset) ──
print("\n  Mengekstraksi fitur dari seluruh dataset...")
X_raw = []
for i, img in enumerate(images):
    X_raw.append(ekstrak_semua_fitur(img))
    if (i + 1) % 30 == 0:
        print(f"  Progress: {i+1}/{len(images)}")

X = np.array(X_raw)
y = np.array(labels)
print(f"[✓] Matriks fitur: {X.shape[0]} sampel × {X.shape[1]} fitur")

# ── Visualisasi Perbandingan Fitur Antar Kelas ──
print("\n  Menampilkan perbandingan fitur antar kelas...")

df_feat = pd.DataFrame(X, columns=FEATURE_NAMES)
df_feat['Kelas'] = [KELAS[l]['nama'] for l in y]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("STEP 4 — PERBANDINGAN FITUR ANTAR KELAS\n"
             "(Distribusi nilai fitur untuk setiap kondisi kesehatan)",
             fontsize=12, fontweight='bold')

fitur_kunci = ['R_mean', 'S_mean', 'V_mean',
               'GLCM_contrast', 'GLCM_energy', 'circularity']
titles_kunci = ['Warna: R_mean\n(Rata-rata Merah)',
                'Warna: S_mean\n(Saturasi HSV)',
                'Warna: V_mean\n(Kecerahan HSV)',
                'GLCM: Contrast\n(Kekasaran Tekstur)',
                'GLCM: Energy\n(Keseragaman Tekstur)',
                'Bentuk: Circularity\n(Kebulatan Telur)']

colors_kelas = [KELAS[k]['warna_ui'] for k in range(N_CLASSES)]
short = ['Sehat', 'Kurang Ca', 'Kurang D3', 'Stres Panas', 'Infeksi', 'Kotor']

for ax, feat, title in zip(axes.flat, fitur_kunci, titles_kunci):
    data_per_kelas = [df_feat[df_feat['Kelas'] == KELAS[k]['nama']][feat].values
                      for k in range(N_CLASSES)]
    bp = ax.boxplot(data_per_kelas, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors_kelas):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticklabels(short, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('Nilai Fitur', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('STEP4_perbandingan_fitur.png', dpi=150, bbox_inches='tight')
plt.show()
print("[✓] Perbandingan fitur disimpan: STEP4_perbandingan_fitur.png")


# ═══════════════════════════════════════════════════════════════════════════════
#   ███████████████  STEP 5 : KLASIFIKASI (MACHINE LEARNING)  ██████████████
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — NORMALISASI & SPLIT DATA

print("\n" + "═" * 65)
print("  STEP 5 — KLASIFIKASI (Machine Learning)")
print("  Data fitur dari Step 4 menjadi input model ML...")
print("═" * 65)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_samples = np.min(np.bincount(y))
if min_samples < 2:
    print("  [!] PERINGATAN: Dataset sangat kecil (ada kelas dengan <2 foto).")
    print("  [!] Stratifikasi dimatikan agar program tetap berjalan.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2 if len(y) >= 5 else 0.5, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

# Jika dataset terlalu kecil sampai test set kosong
if len(X_test) == 0:
    X_test, y_test = X_train, y_train
print(f"\n  Data Training : {X_train.shape[0]} sampel")
print(f"  Data Testing  : {X_test.shape[0]} sampel")
print(f"  Jumlah Fitur  : {X_train.shape[1]} fitur")

# CELL 10 — TRAINING & EVALUASI MODEL

n_neighbors_safe = max(1, min(5, len(X_train)))
models = {
    f'KNN (k={n_neighbors_safe})': KNeighborsClassifier(n_neighbors=n_neighbors_safe),
    'Decision Tree'    : DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest'    : RandomForestClassifier(n_estimators=150, max_depth=8,
                                                random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,
                                                     max_depth=4, random_state=42),
}

hasil_model = {}
print("\n  Training & Evaluasi Model:\n")

for nama, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred   = mdl.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    
    cv_splits = min(5, min_samples)
    if cv_splits < 2:
        cv_score = acc
    else:
        cv_score = cross_val_score(mdl, X_scaled, y, cv=cv_splits, scoring='accuracy').mean()
        
    hasil_model[nama] = {'model': mdl, 'y_pred': y_pred, 'acc': acc, 'cv': cv_score}
    print(f"  {nama:<22} → Test: {acc*100:.1f}%  |  CV: {cv_score*100:.1f}%")

best_name = max(hasil_model, key=lambda k: hasil_model[k]['cv'])
best_mdl  = hasil_model[best_name]['model']
print(f"\n  ✅ Model Terbaik: {best_name}")

# CELL 11 — VISUALISASI PERBANDINGAN AKURASI MODEL

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("STEP 5 — HASIL KLASIFIKASI MACHINE LEARNING",
             fontsize=13, fontweight='bold')

# Grafik akurasi test & CV
names_mdl  = list(hasil_model.keys())
acc_test   = [hasil_model[n]['acc'] * 100 for n in names_mdl]
acc_cv     = [hasil_model[n]['cv']  * 100 for n in names_mdl]
x          = np.arange(len(names_mdl))
width      = 0.35

bars1 = axes[0].bar(x - width/2, acc_test, width, label='Test Accuracy',
                    color='#3498db', edgecolor='black', linewidth=0.7)
bars2 = axes[0].bar(x + width/2, acc_cv,   width, label='Cross-Val (5-fold)',
                    color='#e74c3c', edgecolor='black', linewidth=0.7)
axes[0].set_ylim(0, 115)
axes[0].set_ylabel('Akurasi (%)', fontsize=10)
axes[0].set_title('Perbandingan Akurasi Model', fontsize=10, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names_mdl, rotation=20, ha='right', fontsize=8)
axes[0].legend(fontsize=8)
axes[0].axhline(80, color='gray', ls='--', lw=1.2, label='Threshold 80%')
axes[0].grid(axis='y', alpha=0.3)

for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=7, fontweight='bold')
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=7, fontweight='bold')

# Confusion Matrix (model terbaik)
short_names = ['Sehat', 'Kurang Ca', 'Kurang D3', 'Stres Panas', 'Infeksi', 'Kotor']
cm = confusion_matrix(y_test, hasil_model[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names,
            ax=axes[1], linewidths=0.5, linecolor='white')
axes[1].set_title(f'Confusion Matrix — {best_name}\n'
                  f'Accuracy: {hasil_model[best_name]["acc"]*100:.1f}%',
                  fontsize=10, fontweight='bold')
axes[1].set_xlabel('Prediksi', fontsize=9)
axes[1].set_ylabel('Aktual', fontsize=9)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=7)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=7)

plt.tight_layout()
plt.savefig('STEP5_klasifikasi_hasil.png', dpi=150, bbox_inches='tight')
plt.show()
print("[✓] Hasil klasifikasi disimpan: STEP5_klasifikasi_hasil.png")

# CELL 12 — CLASSIFICATION REPORT

print(f"\n{'='*65}")
print(f"  CLASSIFICATION REPORT — {best_name}")
print(f"{'='*65}")
print(classification_report(
    y_test, hasil_model[best_name]['y_pred'],
    target_names=short_names, digits=3
))


# ═══════════════════════════════════════════════════════════════════════════════
#   ████████████████████  DIAGNOSIS SATU GAMBAR  ████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — FUNGSI DIAGNOSIS LENGKAP (PIPELINE PENUH + VISUALISASI)

def diagnosis_satu_gambar(img_rgb, model=None, judul="Telur Uji", simpan=None):
    """
    Pipeline diagnosis lengkap untuk SATU gambar:
    Menampilkan setiap step pengolahan citra + hasil klasifikasi ML.
    
    Parameter:
        img_rgb : numpy array (H, W, 3) — gambar telur format RGB
        model   : model sklearn yang sudah di-training (default: best_mdl)
        judul   : judul untuk tampilan
        simpan  : path file output (opsional)
    """
    if model is None:
        model = best_mdl

    # ── Step-by-step pengolahan citra ──
    gray, blurred, enhanced, binary, cleaned = preprocess(img_rgb)
    segmented, egg_mask, contour, overlay    = segmentasi(img_rgb, cleaned)

    feats_w = fitur_warna(img_rgb, egg_mask)
    feats_t = fitur_tekstur_glcm(gray, egg_mask)
    feats_b = fitur_bentuk(contour)
    semua   = np.array(feats_w + feats_t + feats_b).reshape(1, -1)

    # ── Prediksi ML ──
    fitur_scaled = scaler.transform(semua)
    pred_kelas   = model.predict(fitur_scaled)[0]
    prob_arr     = model.predict_proba(fitur_scaled)[0] \
                   if hasattr(model, 'predict_proba') else None

    info = KELAS[pred_kelas]

    # ── Figure diagnosis ──
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(f"DIAGNOSIS LENGKAP — {info['ikon']} {info['nama'].upper()}\n"
                 f"[ {judul} ]",
                 fontsize=14, fontweight='bold', color=info['warna_ui'])

    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.50, wspace=0.40)

    # Baris 1: Step-by-step citra
    steps_row1 = [
        (img_rgb,   'Cmap', 'STEP 1\nAkuisisi Citra\n(Original)'),
        (gray,      'gray', 'STEP 2a\nGrayscale'),
        (enhanced,  'gray', 'STEP 2b\nCLAHE\n(Enhanced)'),
        (cleaned,   'gray', 'STEP 2c\nBinary Mask\n(Morfologi)'),
        (overlay,   'Cmap', 'STEP 3\nSegmentasi\n(Kontur Telur)'),
        (segmented, 'Cmap', 'STEP 3\nHasil Segmentasi\n(Telur Terisolasi)'),
    ]
    for col, (img, cmap, title) in enumerate(steps_row1):
        ax = fig.add_subplot(gs[0, col])
        if cmap == 'Cmap':
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=7.5, fontweight='bold', pad=5)
        ax.axis('off')
        # Highlight step terakhir (segmentasi)
        if col == 5:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(info['warna_ui'])
                sp.set_linewidth(3)

    # Baris 2: Fitur + Probabilitas + Hasil
    # [1,0-1] Histogram warna
    ax_hist = fig.add_subplot(gs[1, 0:2])
    if egg_mask is not None:
        for ch, col_c, lbl in zip([0, 1, 2], ['red', 'green', 'blue'], ['R', 'G', 'B']):
            vals = img_rgb[:, :, ch][egg_mask > 0]
            ax_hist.hist(vals, bins=40, color=col_c, alpha=0.5, label=lbl)
    ax_hist.set_title('STEP 4 — Fitur Warna (Histogram RGB)', fontsize=8, fontweight='bold')
    ax_hist.set_xlabel('Intensitas (0–255)', fontsize=7)
    ax_hist.legend(fontsize=7)
    ax_hist.tick_params(labelsize=6)

    # [1,2] GLCM bar
    ax_glcm = fig.add_subplot(gs[1, 2])
    glcm_lbls = ['Energy', 'Contrast', 'Homogen.', 'Corr.', 'Dissim.']
    glcm_vals = feats_t[:5]
    max_g = max(abs(v) for v in glcm_vals) + 1e-6
    glcm_n = [v / max_g for v in glcm_vals]
    colors_g = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    ax_glcm.barh(glcm_lbls, glcm_n, color=colors_g)
    ax_glcm.set_title('STEP 4 — Fitur Tekstur\n(GLCM)', fontsize=8, fontweight='bold')
    ax_glcm.set_xlabel('Normalized', fontsize=7)
    ax_glcm.tick_params(labelsize=6)

    # [1,3] Fitur Bentuk
    ax_shape = fig.add_subplot(gs[1, 3])
    sh_lbls = ['Aspect\nRatio', 'Circular.', 'Solidity', 'Extent']
    sh_vals = feats_b[:4]
    ax_shape.bar(sh_lbls, sh_vals, color=['#1abc9c', '#e74c3c', '#3498db', '#e67e22'])
    ax_shape.set_title('STEP 4 — Fitur Bentuk\n(Geometri)', fontsize=8, fontweight='bold')
    ax_shape.tick_params(labelsize=6)
    ax_shape.set_ylabel('Nilai', fontsize=7)

    # [1,4-5] Probabilitas & Hasil Diagnosis
    ax_prob = fig.add_subplot(gs[1, 4])
    if prob_arr is not None:
        prob_pct = prob_arr * 100
        bar_cols = [KELAS[k]['warna_ui'] for k in range(N_CLASSES)]
        bars_p   = ax_prob.bar(short_names, prob_pct, color=bar_cols,
                               edgecolor='black', linewidth=0.5)
        ax_prob.set_title('STEP 5 — Probabilitas\nPrediksi (ML)', fontsize=8, fontweight='bold')
        ax_prob.set_ylabel('Probabilitas (%)', fontsize=7)
        ax_prob.set_xticklabels(short_names, rotation=25, ha='right', fontsize=6)
        ax_prob.set_ylim(0, 115)
        ax_prob.tick_params(labelsize=6)
        for bar, val in zip(bars_p, prob_pct):
            if val > 3:
                ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                             f'{val:.1f}%', ha='center', fontsize=6, fontweight='bold')
    else:
        ax_prob.text(0.5, 0.5, 'Probabilitas\ntidak tersedia',
                     ha='center', va='center', transform=ax_prob.transAxes)
        ax_prob.axis('off')

    # [1,5] Panel Hasil Diagnosis
    ax_hasil = fig.add_subplot(gs[1, 5])
    ax_hasil.set_facecolor(info['warna_ui'] + '22')  # warna transparan
    ax_hasil.axis('off')

    prob_txt = f"{prob_arr[pred_kelas]*100:.1f}%" if prob_arr is not None else "—"
    diagnosis_text = (
        f"╔══ HASIL DIAGNOSIS ══╗\n\n"
        f"  {info['ikon']}  {info['nama']}\n\n"
        f"  Keyakinan: {prob_txt}\n"
        f"  Bahaya   : {info['bahaya']}\n\n"
        f"══ TANDA TELUR ══\n"
        f"{info['tanda_telur'][:60]}...\n\n"
        f"══ PENYEBAB ══\n"
        f"{info['penyebab'][:60]}...\n\n"
        f"══ REKOMENDASI ══\n"
    )
    for i, rec in enumerate(info['rekomendasi'], 1):
        diagnosis_text += f"  {i}. {rec[:45]}...\n"

    ax_hasil.text(0.05, 0.95, diagnosis_text, ha='left', va='top',
                  fontsize=7, transform=ax_hasil.transAxes,
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white', alpha=0.9,
                             edgecolor=info['warna_ui'], linewidth=2))

    if simpan:
        plt.savefig(simpan, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'═'*65}")
    print(f"  {info['ikon']}  DIAGNOSIS: {info['nama'].upper()}")
    print(f"  Bahaya      : {info['bahaya']}")
    print(f"  Probabilitas: {prob_txt}")
    print(f"{'─'*65}")
    print(f"  Tanda Telur : {info['tanda_telur']}")
    print(f"  Penyebab    : {info['penyebab']}")
    print(f"  Rekomendasi :")
    for r in info['rekomendasi']:
        print(f"    → {r}")
    print(f"{'═'*65}")


# ── Demo: Diagnosis 3 Sampel ──
print("\n" + "═" * 65)
print("  DEMO DIAGNOSIS LENGKAP (Pipeline Penuh)")
print("═" * 65)

demo_kelas = [0, 4, 5]  # Sehat, Infeksi, Kotor
for k in demo_kelas:
    idx  = labels.index(k)
    info = KELAS[k]
    print(f"\n→ Mendiagnosis: {info['ikon']} {info['nama']}")
    diagnosis_satu_gambar(
        images[idx],
        judul=f"Kelas {k}: {info['nama']}",
        simpan=f"DIAGNOSIS_kelas{k}.png"
    )

print("\n[✓] Semua STEP selesai!")
print("\n📁 File yang dihasilkan:")
print("   STEP1_akuisisi_citra.png")
print("   STEP2_preprocessing_kelas0-5.png")
print("   STEP3_segmentasi_kelas0-5.png")
print("   STEP4_fitur_kelas0-5.png")
print("   STEP4_perbandingan_fitur.png")
print("   STEP5_klasifikasi_hasil.png")
print("   DIAGNOSIS_kelas0.png, DIAGNOSIS_kelas4.png, DIAGNOSIS_kelas5.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — UPLOAD GAMBAR SENDIRI (GOOGLE COLAB)
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════
#  Untuk menguji dengan FOTO TELUR ASLI:
#
#  from google.colab import files
#  import PIL.Image
#  uploaded = files.upload()                     # pilih foto telur
#  img_path = list(uploaded.keys())[0]
#  img_asli = np.array(PIL.Image.open(img_path).convert('RGB').resize((300,300)))
#  diagnosis_satu_gambar(img_asli, judul="Foto Telur Asli")
# ════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════╗
║   RINGKASAN PIPELINE PENGOLAHAN CITRA                        ║
╠══════════════════════════════════════════════════════════════╣
║  STEP 1 : Akuisisi Citra  → Gambar asli / foto telur         ║
║  STEP 2 : Pre-processing  → Grayscale, Blur, CLAHE, Binary   ║
║  STEP 3 : Segmentasi      → Kontur, Masking, Isolasi Telur   ║
║  STEP 4 : Ekstraksi Fitur → RGB/HSV/LAB + GLCM + Geometri   ║
║  STEP 5 : Klasifikasi ML  → KNN/DT/RF/GB → Diagnosis Kelas  ║
╠══════════════════════════════════════════════════════════════╣
║  6 KELAS OUTPUT:                                             ║
║  ✅ Kelas 0 : Sehat / Normal                                 ║
║  🦴 Kelas 1 : Kekurangan Kalsium                            ║
║  ☀️  Kelas 2 : Kekurangan Vitamin D3                         ║
║  🌡️  Kelas 3 : Stres Panas (Heat Stress)                     ║
║  🦠 Kelas 4 : Infeksi (Newcastle / IB)                      ║
║  🧫 Kelas 5 : Kebersihan Buruk / Kontaminasi                ║
╚══════════════════════════════════════════════════════════════╝
""")
