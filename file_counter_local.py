"""
Count image/PDF files per immediate subfolder under a local INPUT_DIR.
Same output style as file_counter.py (print + CSV). No Azure.
"""
import csv
import os
import sys
from pathlib import Path

# ============================================================
# LOCAL INPUT
# ============================================================

INPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")

# ============================================================
# FILE EXTENSIONS TO COUNT
# ============================================================

EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
    ".jp2",
    ".j2k",
    ".pdf",
)

# ============================================================
# CSV OUTPUT
# ============================================================

CSV_PATH = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "jpg_counts_local.csv",
)


print()
print("=" * 70)
print("Scanning local folder")
print("=" * 70)
print(f"INPUT: {INPUT_DIR}")
print("=" * 70)

if not INPUT_DIR.is_dir():
    print("\nERROR: INPUT_DIR does not exist or is not a folder:")
    print(f"  {INPUT_DIR}")
    input("\nPress Enter to close...")
    sys.exit(1)

folder_counts = {}
total_files_scanned = 0
total_files_counted = 0

try:
    chart_folders = sorted(
        (p for p in INPUT_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower(),
    )

    for folder in chart_folders:
        count = 0
        for entry in folder.iterdir():
            if not entry.is_file():
                continue
            total_files_scanned += 1
            if entry.suffix.lower() in EXTENSIONS:
                count += 1
                total_files_counted += 1
        folder_counts[folder.name] = count

except Exception as e:
    print("\nERROR while reading local folder:")
    print(e)
    input("\nPress Enter to close...")
    sys.exit(1)


print()
print("=" * 70)
print("SCAN RESULTS")
print("=" * 70)
print(f"Total files scanned under folders : {total_files_scanned}")
print(f"Total files counted               : {total_files_counted}")
print(f"Total folders found               : {len(folder_counts)}")
print()

if not folder_counts:
    print("NO FOLDERS WERE FOUND.")
    print()
    print("Check INPUT_DIR.")
    print()
else:
    rows = sorted(folder_counts.items(), key=lambda x: x[0].lower())

    print(f"{'Folder Name':<50} {'Count':>10}")
    print("-" * 50, "-" * 10)

    for folder_name, count in rows:
        print(f"{folder_name:<50} {count:>10}")

    print("-" * 50, "-" * 10)

    total = sum(count for _, count in rows)
    print(f"{'Total':<50} {total:>10}")

    try:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Folder Name", "Count"])
            for folder_name, count in rows:
                writer.writerow([folder_name, count])
            writer.writerow(["Total", total])

        print()
        print("CSV saved to:")
        print(CSV_PATH)

    except Exception as e:
        print()
        print("ERROR saving CSV:")
        print(e)

print()
input("Press Enter to close...")
