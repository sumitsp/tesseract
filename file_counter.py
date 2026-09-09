import csv
import os
import sys
import time
import traceback

from azure.core.exceptions import AzureError, HttpResponseError, ServiceRequestError
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ExponentialRetry


# ============================================================
# AZURE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"

# CHANGE THIS to your actual container name (lowercase, hyphens only)
CONTAINER_NAME = "YOUR_CONTAINER_NAME"

# Make sure this EXACTLY matches the folder structure in Azure
PREFIX = "Run1/Batch1/DEID_PNGs/"


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
    "jpg_counts_batch1.csv",
)


# ============================================================
# AZURE CONNECTION
# ============================================================

print("Connecting to Azure Blob Storage...", flush=True)

if CONTAINER_NAME in ("", "YOUR_CONTAINER_NAME"):
    print("\nERROR: Set CONTAINER_NAME to your real container name.")
    print("Placeholder YOUR_CONTAINER_NAME is invalid in Azure.")
    input("\nPress Enter to close...")
    sys.exit(1)

ACCOUNT_URL = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"

# More retries / longer waits — helps with "Unable to stream download"
# when VPN/proxy drops the connection mid-response.
RETRY = ExponentialRetry(
    initial_backoff=2,
    increment_base=2,
    retry_total=12,
    retry_to_secondary=True,
)


def make_credential():
    # Prefer `az login` token first (most reliable on corporate laptops).
    try:
        cred = AzureCliCredential()
        # Force a quick token fetch so we fail early with a clear message.
        cred.get_token("https://storage.azure.com/.default")
        print("Auth: AzureCliCredential (az login)", flush=True)
        return cred
    except Exception as exc:
        print(f"AzureCliCredential failed ({exc}); falling back to DefaultAzureCredential...", flush=True)
        return DefaultAzureCredential(exclude_interactive_browser_credential=False)


try:
    credential = make_credential()

    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
        retry_policy=RETRY,
        connection_timeout=120,
        read_timeout=300,
    )

    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Quick connectivity check before the full scan.
    props = container_client.get_container_properties()
    print(f"Azure connection OK. Container: {props.name}", flush=True)

except Exception as e:
    print("\nERROR connecting to Azure:")
    print(e)
    print("\n--- full traceback ---")
    traceback.print_exc()
    print(
        "\nChecks:\n"
        "  1. az login\n"
        "  2. CONTAINER_NAME is correct\n"
        "  3. VPN connected\n"
        f"  4. Test: az storage blob list --account-name {STORAGE_ACCOUNT} "
        f"--container-name {CONTAINER_NAME} --prefix \"{PREFIX}\" "
        "--auth-mode login --num-results 5 -o table"
    )
    input("\nPress Enter to close...")
    sys.exit(1)


# ============================================================
# SCAN BLOB STORAGE
# ============================================================

print()
print("=" * 70)
print("Scanning Azure Blob")
print("=" * 70)
print(f"Storage Account : {STORAGE_ACCOUNT}")
print(f"Container       : {CONTAINER_NAME}")
print(f"Prefix          : {PREFIX}")
print("=" * 70)


folder_counts = {}

total_blobs_found = 0
total_files_counted = 0


def iter_blobs_with_retry(client, prefix, max_attempts=8):
    """List blobs page-by-page; retry a page if the stream drops."""
    continuation = None
    while True:
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                pages = client.list_blobs(
                    name_starts_with=prefix,
                    results_per_page=500,
                ).by_page(continuation_token=continuation)

                page = next(pages)
                for blob in page:
                    yield blob
                continuation = pages.continuation_token
                break
            except (ServiceRequestError, HttpResponseError, AzureError, OSError) as exc:
                last_error = exc
                wait = min(60, 2 ** attempt)
                print(
                    f"\nStream/list interrupted (attempt {attempt}/{max_attempts}): {exc}",
                    flush=True,
                )
                print(f"Retrying in {wait}s...", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"Gave up listing blobs after {max_attempts} attempts. Last error: {last_error}"
            ) from last_error

        if not continuation:
            return


try:
    for blob in iter_blobs_with_retry(container_client, PREFIX):
        total_blobs_found += 1

        if total_blobs_found % 1000 == 0:
            print(f"  ... scanned {total_blobs_found} blobs", flush=True)

        blob_name = blob.name
        relative_path = blob_name[len(PREFIX) :]
        parts = relative_path.split("/")

        # Ignore files directly inside DEID_PNGs
        if len(parts) < 2:
            continue

        folder_name = parts[0]
        filename = parts[-1]

        if not filename.lower().endswith(EXTENSIONS):
            continue

        folder_counts[folder_name] = folder_counts.get(folder_name, 0) + 1
        total_files_counted += 1

except Exception as e:
    print("\nERROR while reading blobs:")
    print(e)
    print("\n--- full traceback ---")
    traceback.print_exc()
    input("\nPress Enter to close...")
    sys.exit(1)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("SCAN RESULTS")
print("=" * 70)

print(f"Total blobs found under prefix : {total_blobs_found}")
print(f"Total files counted            : {total_files_counted}")
print(f"Total folders found            : {len(folder_counts)}")
print()


if not folder_counts:

    print("NO FOLDERS / FILES WERE COUNTED.")
    print()
    print("This usually means one of these is wrong:")
    print()
    print("1. CONTAINER_NAME")
    print("2. PREFIX")
    print("3. Blob folder structure")
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
        print(f"CSV saved to:")
        print(CSV_PATH)

    except Exception as e:
        print()
        print("ERROR saving CSV:")
        print(e)


print()
input("Press Enter to close...")
