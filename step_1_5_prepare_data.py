import os
import shutil
import random

# --- DATA SOURCE PATHS ---
EXTRACTED_REAL = r"D:\DeepGuard\Dataset\Real"
EXTRACTED_FAKE = r"D:\DeepGuard\Dataset\Fake"

KAGGLE_REAL = r"D:\DeepGuard\Forensic++\archive\my_real_vs_ai_dataset\my_real_vs_ai_dataset\real"
KAGGLE_FAKE = r"D:\DeepGuard\Forensic++\archive\my_real_vs_ai_dataset\my_real_vs_ai_dataset\ai_images"

# --- DESTINATION CONFIGURATION ---
DEST_BASE = r"D:\DeepGuard\Mini_Dataset"
EXTRA_KAGGLE = 500  # Number of external images to augment
VAL_SPLIT = 0.2     # 20% validation split

def mix_and_split(ext_dir, kaggle_dir, category):
    print(f"\n⏳ Preparing {category} Data...")

    # 1. Load Extracted Images
    ext_imgs = [os.path.join(ext_dir, f) for f in os.listdir(ext_dir) if f.endswith('.jpg')]

    # 2. Load and Shuffle Kaggle Images
    kag_all = [f for f in os.listdir(kaggle_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(kag_all)
    kag_imgs = [os.path.join(kaggle_dir, f) for f in kag_all[:EXTRA_KAGGLE]]

    # 3. Mix Datasets & Shuffle
    all_imgs = ext_imgs + kag_imgs
    random.shuffle(all_imgs)

    val_count = int(len(all_imgs) * VAL_SPLIT)
    train_count = len(all_imgs) - val_count

    train_imgs = all_imgs[:train_count]
    val_imgs = all_imgs[train_count:]

    # Create Directories
    os.makedirs(os.path.join(DEST_BASE, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(DEST_BASE, 'val', category), exist_ok=True)

    # Copy Files to Destination
    for img in train_imgs:
        shutil.copy(img, os.path.join(DEST_BASE, 'train', category, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy(img, os.path.join(DEST_BASE, 'val', category, os.path.basename(img)))

    print(f"✅ {category} Ready! Total: {len(all_imgs)} (Extracted: {len(ext_imgs)} | Kaggle: {len(kag_imgs)})")


# --- EXECUTION ---
print("🚀 INITIATING DATASET OPTIMIZATION...")
if os.path.exists(DEST_BASE):
    shutil.rmtree(DEST_BASE)

mix_and_split(EXTRACTED_REAL, KAGGLE_REAL, "Real")
mix_and_split(EXTRACTED_FAKE, KAGGLE_FAKE, "Fake")
print("\n🎉 PROCESS COMPLETE! Data split into Train & Val. Proceed to step_2.")