"""
Priprema dataseta za treniranje
 
Koraci:
  1. Skaliranje svih slika na 224x224px
  2. Augmentacija (rotacija, flip, brightness, zoom)
  3. Provera balansa klasa
  4. Podela na train / val / test (70% / 15% / 15%)
 
Ulaz:  dataset/edible/*   i   dataset/inedible/*
Izlaz: dataset_prepared/
           train/
               edible/
               inedible/
           val/
               edible/
               inedible/
           test/
               edible/
               inedible/
 
"""
 
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
 
# Konfiguracija
 
INPUT_DIR      = Path("dataset")           # izlaz iz skripte za skupljanje dataseta
OUTPUT_DIR     = Path("dataset_prepared")  # ovde ide pripremljeni dataset
 
IMG_SIZE       = (224, 224)   # standardna velicina za CNN (MobileNet, ResNet)
TARGET_PER_CLASS = 500        # cilj po klasi nakon augmentacije
 
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15         # ostatak
 
RANDOM_SEED    = 42           # za reproduktibilnost
 
CATEGORIES     = ["edible", "inedible"]
 
# Augmentacija
 
def augment_image(img: Image.Image) -> list[Image.Image]:
    """
    Generise augmentovane verzije jedne slike.
    Vraca listu novih PIL slika (bez originala).
    """
    augmented = []
 
    # Horizontalni flip
    augmented.append(ImageOps.mirror(img))
 
    # Vertikalni flip (biljke u prirodi — ponekad korisno)
    augmented.append(ImageOps.flip(img))
 
    # Rotacije
    for angle in [15, 30, 45, 90, 180, 270, -15, -30]:
        rotated = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        augmented.append(rotated)
 
    # Brightness varijacije
    for factor in [0.7, 0.85, 1.15, 1.3]:
        bright = ImageEnhance.Brightness(img).enhance(factor)
        augmented.append(bright)
 
    # Kontrast varijacije
    for factor in [0.8, 1.2]:
        contrast = ImageEnhance.Contrast(img).enhance(factor)
        augmented.append(contrast)
 
    # Zoom (crop + resize)
    w, h = img.size
    for crop_ratio in [0.85, 0.75]:
        cw, ch = int(w * crop_ratio), int(h * crop_ratio)
        left = (w - cw) // 2
        top  = (h - ch) // 2
        cropped = img.crop((left, top, left + cw, top + ch))
        augmented.append(cropped.resize(IMG_SIZE, Image.LANCZOS))
 
    return augmented
 
 
def load_and_resize(path: Path) -> Image.Image | None:
    """Ucitava sliku i skalira je na IMG_SIZE."""
    try:
        img = Image.open(path).convert("RGB")
        return img.resize(IMG_SIZE, Image.LANCZOS)
    except Exception:
        return None
 
 
# Korak 1: Prikupi sve slike po kategoriji
 
def collect_images(category: str) -> list[Path]:
    """Vraca sve .jpg slike za datu kategoriju (edible/inedible)."""
    cat_dir = INPUT_DIR / category
    if not cat_dir.exists():
        print(f"  Folder ne postoji: {cat_dir}")
        return []
 
    images = list(cat_dir.rglob("*.jpg")) + list(cat_dir.rglob("*.jpeg")) + \
             list(cat_dir.rglob("*.png"))
    return images
 
 
# Korak 2: Augmentacija do TARGET_PER_CLASS
 
def prepare_category(category: str, output_pool: Path) -> list[Path]:
    """
    Ucitava originalne slike, augmentuje do TARGET_PER_CLASS,
    cuva u privremeni pool folder.
    Vraca listu putanja svih spremnih slika.
    """
    output_pool.mkdir(parents=True, exist_ok=True)
    images = collect_images(category)
 
    if not images:
        print(f"  Nema slika za kategoriju: {category}")
        return []
 
    print(f"\n  [{category.upper()}] — {len(images)} originalnih slika")
 
    all_paths = []
    aug_count = 0
    idx = 0
 
    with tqdm(total=TARGET_PER_CLASS, desc=f"  Priprema {category}", unit="slika") as pbar:
        # Prvo kopiraj originale (skalirane)
        for src in images:
            img = load_and_resize(src)
            if img is None:
                continue
            dest = output_pool / f"orig_{idx:05d}.jpg"
            img.save(dest, "JPEG", quality=90)
            all_paths.append(dest)
            idx += 1
            pbar.update(1)
            if idx >= TARGET_PER_CLASS:
                break
 
        # Augmentacija ako treba vise
        if idx < TARGET_PER_CLASS:
            random.shuffle(images)
            for src in images:
                if idx >= TARGET_PER_CLASS:
                    break
                img = load_and_resize(src)
                if img is None:
                    continue
                for aug_img in augment_image(img):
                    if idx >= TARGET_PER_CLASS:
                        break
                    dest = output_pool / f"aug_{idx:05d}.jpg"
                    aug_img_resized = aug_img.resize(IMG_SIZE, Image.LANCZOS)
                    aug_img_resized.save(dest, "JPEG", quality=90)
                    all_paths.append(dest)
                    aug_count += 1
                    idx += 1
                    pbar.update(1)
 
    print(f"  Originali: {idx - aug_count} | Augmentovano: {aug_count} | Ukupno: {idx}")
    return all_paths
 
 
# Korak 3: Podela na train/val/test
 
def split_and_copy(paths: list[Path], category: str) -> dict:
    """
    Deli listu slika na train/val/test i kopira ih u OUTPUT_DIR.
    Vraca brojeve po splitu.
    """
    random.shuffle(paths)
 
    train_val, test = train_test_split(paths, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=RANDOM_SEED)
 
    splits = {"train": train, "val": val, "test": test}
    counts = {}
 
    for split_name, split_paths in splits.items():
        dest_dir = OUTPUT_DIR / split_name / category
        dest_dir.mkdir(parents=True, exist_ok=True)
 
        for i, src in enumerate(split_paths):
            dest = dest_dir / f"{category}_{split_name}_{i:05d}.jpg"
            shutil.copy2(src, dest)
 
        counts[split_name] = len(split_paths)
        print(f"    {split_name:<6}: {len(split_paths)} slika")
 
    return counts
 
 
# Korak 4: Provera balansa
 
def check_balance(stats: dict) -> None:
    """Stampa upozorenje ako su klase neuravnotezene."""
    print("\n" + "=" * 50)
    print("PROVERA BALANSA KLASA")
    print("=" * 50)
 
    for split in ["train", "val", "test"]:
        counts = {cat: stats[cat][split] for cat in CATEGORIES if cat in stats}
        total = sum(counts.values())
        print(f"\n  [{split.upper()}] — ukupno: {total}")
        for cat, count in counts.items():
            ratio = count / total * 100 if total > 0 else 0
            bar = "#" * int(ratio / 5)
            print(f"    {cat:<12} {count:>4} slika  {ratio:5.1f}%  {bar}")
 
        values = list(counts.values())
        if max(values) / max(min(values), 1) > 1.5:
            print(f"  UPOZORENJE: Neuravnotezene klase! Razmotri weighted loss u treniranju.")
        else:
            print(f"  OK: Klase su uravnotezene")
 
 
# Main
 
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
 
    print("KT2 — Priprema dataseta")
    print(f"   Ulaz:  {INPUT_DIR.resolve()}")
    print(f"   Izlaz: {OUTPUT_DIR.resolve()}")
    print(f"   Cilj:  {TARGET_PER_CLASS} slika po klasi")
    print(f"   Split: {int(TRAIN_RATIO*100)}% train / {int(VAL_RATIO*100)}% val / {int(TEST_RATIO*100)}% test\n")
 
    if OUTPUT_DIR.exists():
        print("  Brisem stari dataset_prepared folder...")
        shutil.rmtree(OUTPUT_DIR)
 
    pool_dir = Path("_pool_temp")
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
 
    stats = {}
 
    for category in CATEGORIES:
        print(f"\n{'='*50}")
        print(f"  Kategorija: {category.upper()}")
        print(f"{'='*50}")
 
        pool = pool_dir / category
        all_paths = prepare_category(category, pool)
 
        if not all_paths:
            continue
 
        print(f"\n  Podela na train/val/test:")
        counts = split_and_copy(all_paths, category)
        stats[category] = counts
 
    check_balance(stats)
 
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
 
    print("\n" + "=" * 50)
    print("KT2 zavrsena!")
    print(f"   Dataset spreman u: {OUTPUT_DIR.resolve()}")
    print("""
   Sledeci korak — u Keras/TensorFlow:
 
   from tensorflow.keras.utils import image_dataset_from_directory
 
   train_ds = image_dataset_from_directory(
       'dataset_prepared/train',
       image_size=(224, 224),
       batch_size=32
   )
   val_ds = image_dataset_from_directory(
       'dataset_prepared/val',
       image_size=(224, 224),
       batch_size=32
   )
    """)
 
 
if __name__ == "__main__":
    main()