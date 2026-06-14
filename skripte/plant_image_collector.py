"""
Preuzima slike 6 vrsta sa GBIF-a i organizuje ih u folder strukturu
pogodnu za Keras/TensorFlow treniranje.
 
Filteri:
  - Odbacuje herbarske snimke (po URL-u, publisher-u i basis of record)
  - Odbacuje slike ispod minimalne rezolucije
  - Odbacuje slike sa pretezno belom/sivom pozadinom (herbari)
 
Struktura:
    dataset/
    edible/
        allium_ursinum/
        allium_vineale/
        allium_sphaerocephalon/
    inedible/
        colchicum_autumnale/
        arum_maculatum/
        convallaria_majalis/
 
"""
 
import time
import warnings
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
 
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
 
# Konfiguracija
 
SPECIES = {
    "edible": {
        "allium_ursinum":         "Allium ursinum",
        "allium_vineale":         "Allium vineale",
        "allium_sphaerocephalon": "Allium sphaerocephalon",
    },
    "inedible": {
        "colchicum_autumnale":    "Colchicum autumnale",
        "arum_maculatum":         "Arum maculatum",
        "convallaria_majalis":    "Convallaria majalis",
    }
}
 
TARGET_PER_SPECIES = 200   # Cilj po vrsti
MIN_SIZE_PX        = 224   # Minimalna dimenzija slike
MAX_SIZE_PX        = 4000  # Maksimalna dimenzija (vece su skoro uvek herbari)
OUTPUT_DIR         = Path("dataset")
DELAY_SECONDS      = 0.3
 
GBIF_API = "https://api.gbif.org/v1"
 
# Kljucne reci u URL-u koje ukazuju na herbarski snimak
HERBARIUM_URL_KEYWORDS = [
    "herbarium", "herbar", "specimen", "specimens",
    "kew.org", "jstor.org", "naturalis.nl", "nhm.ac.uk",
    "botanicalcollections", "museum", "mnhn.fr",
    "fieldmuseum", "inaturalist" # iNaturalist je ok ali proveravamo posebno
]
 
# Izdavaci ciji su zapisi skoro uvek herbari
HERBARIUM_PUBLISHERS = [
    "herbarium", "natural history museum", "botanischer garten",
    "royal botanic", "kew", "smithsonian", "museum national"
]
 
# Basis of record koji garantuje herbar
HERBARIUM_BASIS = [
    "PRESERVED_SPECIMEN",
    "FOSSIL_SPECIMEN",
    "LIVING_SPECIMEN"  # zive kulture iz staklenika, ne iz prirode
]
 
# Filteri
 
def is_herbarium_by_metadata(occurrence: dict) -> bool:
    """Proverava metadata occurrence zapisa — da li je herbar?"""
    # Basis of record
    basis = occurrence.get("basisOfRecord", "")
    if basis in HERBARIUM_BASIS:
        return True
 
    # Publisher naziv
    publisher = occurrence.get("datasetName", "").lower()
    if any(term in publisher for term in HERBARIUM_PUBLISHERS):
        return True
 
    return False
 
 
def is_herbarium_by_url(url: str) -> bool:
    """Proverava URL slike — da li ukazuje na herbar?"""
    url_lower = url.lower()
    # iNaturalist je ok
    if "inaturalist" in url_lower:
        return False
    return any(term in url_lower for term in HERBARIUM_URL_KEYWORDS)
 
 
def is_herbarium_by_pixels(img: Image.Image) -> bool:
    """
    Analizira piksele — herbarske slike imaju pretezno belu/krem pozadinu.
    Ako je >60% slike svetlija od praga, verovatno je herbar.
    """
    # Smanji za brzinu analize
    small = img.resize((100, 100)).convert("RGB")
    arr = np.array(small)
 
    # Svetli pikseli: sva tri kanala > 200 (bela/krem pozadina)
    bright_mask = np.all(arr > 200, axis=2)
    bright_ratio = bright_mask.sum() / (100 * 100)
 
    return bright_ratio > 0.60
 
 
def is_valid_image(img_bytes: bytes) -> tuple[bool, Image.Image | None, str]:
    """
    Kompletna validacija slike.
    Vraca (validna, PIL_image, razlog_odbijanja).
    """
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
 
        if w < MIN_SIZE_PX or h < MIN_SIZE_PX:
            return False, None, f"premala ({w}x{h})"
 
        if w > MAX_SIZE_PX or h > MAX_SIZE_PX:
            return False, None, f"prevelika ({w}x{h}) — verovatno herbar"
 
        img_rgb = img.convert("RGB")
 
        if is_herbarium_by_pixels(img_rgb):
            return False, None, "bela pozadina (herbar)"
 
        return True, img_rgb, ""
 
    except Exception as e:
        return False, None, f"greška pri čitanju"
 
 
# GBIF funkcije
 
def get_species_key(species_name: str) -> int | None:
    """Pronalazi GBIF taxon key za dato latinsko ime vrste."""
    resp = requests.get(
        f"{GBIF_API}/species/match",
        params={"name": species_name, "rank": "SPECIES"},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
 
    if data.get("matchType") == "NONE":
        print(f"  Nije pronadjena vrsta: {species_name}")
        return None
 
    key = data.get("usageKey") or data.get("speciesKey")
    print(f"  OK {species_name} -> taxon key: {key}")
    return key
 
 
def fetch_occurrences(species_key: int, limit: int) -> list[dict]:
    """
    Prikuplja GBIF occurrence zapise sa fotografijama iz prirode.
    Koristi basisOfRecord=HUMAN_OBSERVATION da filtrira herbare na nivou API-ja.
    """
    occurrences = []
    offset = 0
    page_size = 100
 
    while len(occurrences) < limit:
        params = {
            "taxonKey":       species_key,
            "mediaType":      "StillImage",
            "basisOfRecord":  "HUMAN_OBSERVATION",  # kljucni filter
            "limit":          page_size,
            "offset":         offset,
        }
        resp = requests.get(f"{GBIF_API}/occurrence/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
 
        results = data.get("results", [])
        if not results:
            break
 
        occurrences.extend(results)
        offset += page_size
 
        if data.get("endOfRecords", True):
            break
 
        time.sleep(DELAY_SECONDS)
 
    return occurrences
 
 
def extract_image_urls(occurrences: list[dict]) -> list[str]:
    """Izvlaci URL-ove slika, uz metadata filter za herbare."""
    urls = []
    skipped_meta = 0
 
    for occ in occurrences:
        if is_herbarium_by_metadata(occ):
            skipped_meta += 1
            continue
 
        for media in occ.get("media", []):
            if media.get("type") == "StillImage":
                url = media.get("identifier") or media.get("references")
                if not url or not url.startswith("http"):
                    continue
                if is_herbarium_by_url(url):
                    skipped_meta += 1
                    continue
                urls.append(url)
                break  # jedna slika po occurrence
 
    if skipped_meta:
        print(f"  Preskoceno {skipped_meta} herbarskih zapisa (metadata filter)")
    return urls
 
 
# Preuzimanje
 
def download_images(urls: list[str], save_dir: Path, target: int) -> int:
    """Preuzima i validira slike, cuva samo slike iz prirode."""
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped_pixel = 0
    img_index = len(list(save_dir.glob("*.jpg")))  # nastavi od prethodnog broja
 
    with tqdm(total=target, desc=f"  {save_dir.name}", unit="slika",
              initial=img_index if img_index < target else 0) as pbar:
 
        if img_index >= target:
            return img_index
 
        for url in urls:
            if downloaded + img_index >= target:
                break
 
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
 
                valid, img_rgb, reason = is_valid_image(resp.content)
 
                if not valid:
                    if "herbar" in reason or "pozadina" in reason:
                        skipped_pixel += 1
                    continue
 
                filename = save_dir / f"{save_dir.name}_{img_index + downloaded:04d}.jpg"
                img_rgb.save(filename, "JPEG", quality=90)
                downloaded += 1
                pbar.update(1)
                time.sleep(DELAY_SECONDS)
 
            except Exception:
                continue
 
    if skipped_pixel:
        print(f"  Odbaceno {skipped_pixel} slika (bela pozadina — pixel filter)")
 
    return downloaded + img_index
 
 
# Rezime
 
def print_summary(stats: dict) -> None:
    print("\n" + "=" * 50)
    print(" REZIME PREUZIMANJA")
    print("=" * 50)
    total = 0
    for category, species_stats in stats.items():
        print(f"\n  [{category.upper()}]")
        for folder, count in species_stats.items():
            status = "OK" if count >= TARGET_PER_SPECIES * 0.8 else "UPOZORENJE"
            print(f"    {status} {folder:<32} {count} slika")
            total += count
    print(f"\n  UKUPNO: {total} slika")
    print("=" * 50)
 
 
# Main
 
def main():
    print("GBIF Image Downloader v2 — samo slike iz prirode")
    print(f"   Cilj: {TARGET_PER_SPECIES} slika po vrsti")
    print(f"   Filteri: API (HUMAN_OBSERVATION) + URL + pixel analiza\n")
 
    stats = {"edible": {}, "inedible": {}}
 
    for category, species_dict in SPECIES.items():
        print(f"\n{'='*50}")
        print(f"  Kategorija: {category.upper()}")
        print(f"{'='*50}")
 
        for folder_name, latin_name in species_dict.items():
            print(f"\n> {latin_name}")
 
            species_key = get_species_key(latin_name)
            if not species_key:
                stats[category][folder_name] = 0
                continue
 
            print(f"  Preuzimam occurrence zapise (HUMAN_OBSERVATION)...")
            occurrences = fetch_occurrences(species_key, limit=TARGET_PER_SPECIES * 4)
            print(f"  Pronadjeno {len(occurrences)} zapisa")
 
            urls = extract_image_urls(occurrences)
            print(f"  Proslo sve filtere: {len(urls)} URL-ova")
 
            if not urls:
                print(f"  Nema slika za {latin_name} — pokusaj bez filtera")
                stats[category][folder_name] = 0
                continue
 
            save_dir = OUTPUT_DIR / category / folder_name
            count = download_images(urls, save_dir, TARGET_PER_SPECIES)
            stats[category][folder_name] = count
 
    print_summary(stats)
 
    print("\nNAPOMENE:")
    for category, species_stats in stats.items():
        for folder, count in species_stats.items():
            if count < 100:
                print(f"  {folder}: samo {count} slika — potrebna augmentacija u KT2")
            elif count < TARGET_PER_SPECIES * 0.8:
                print(f"  {folder}: {count} slika — dovoljno, ali mozes potraziti jos")
 
    print("\nPreuzimanje zavrseno!")
    print(f"   Dataset: {OUTPUT_DIR.resolve()}")
    print("   Sledeci korak (KT2): augmentacija i podela train/val/test\n")
 
 
if __name__ == "__main__":
    main()