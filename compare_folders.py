"""
Compare chart subfolders: Azure Blob (parent) vs local folder (child).

Lists immediate folder names under PREFIX on blob, then prints which of those
are missing from the local child directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ExponentialRetry

# ============================================================
# AZURE CONFIGURATION (same style as file_counter.py)
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

# Local folder to compare against blob parent
LOCAL_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text")


def log(message: str) -> None:
    print(message, flush=True)


def make_credential():
    try:
        cred = AzureCliCredential()
        cred.get_token("https://storage.azure.com/.default")
        log("Auth: AzureCliCredential (az login)")
        return cred
    except Exception as exc:
        log(f"AzureCliCredential failed ({exc}); using DefaultAzureCredential...")
        return DefaultAzureCredential(exclude_interactive_browser_credential=False)


def blob_subfolder_names(
    storage_account: str,
    container_name: str,
    prefix: str,
) -> set[str]:
    """Immediate folder names under PREFIX on blob (FolderA/... -> FolderA)."""
    if container_name in ("", "YOUR_CONTAINER_NAME"):
        raise SystemExit("Set CONTAINER_NAME to your real container name.")

    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    account_url = f"https://{storage_account}.blob.core.windows.net"
    client = BlobServiceClient(
        account_url=account_url,
        credential=make_credential(),
        retry_policy=ExponentialRetry(retry_total=8, initial_backoff=1),
        connection_timeout=60,
        read_timeout=180,
    )
    container = client.get_container_client(container_name)

    log("=" * 70)
    log("Scanning Azure Blob for subfolders")
    log("=" * 70)
    log(f"Storage Account : {storage_account}")
    log(f"Container       : {container_name}")
    log(f"Prefix          : {prefix}")
    log("=" * 70)

    folders: set[str] = set()
    for blob in container.list_blobs(name_starts_with=prefix):
        relative = blob.name[len(prefix) :]
        parts = relative.split("/")
        if len(parts) < 2:
            continue
        folders.add(parts[0])

    return folders


def local_subfolder_names(folder: Path) -> set[str]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    return {p.name for p in folder.iterdir() if p.is_dir()}


def main() -> None:
    local_dir = LOCAL_DIR
    log(f"LOCAL: {local_dir}")

    parent_folders = blob_subfolder_names(STORAGE_ACCOUNT, CONTAINER_NAME, PREFIX)
    child_folders = local_subfolder_names(local_dir.resolve())

    missing = sorted(parent_folders - child_folders, key=str.lower)
    extra = sorted(child_folders - parent_folders, key=str.lower)

    log("")
    log(f"Blob parent folders : {len(parent_folders)}")
    log(f"Local child folders : {len(child_folders)}")
    log(f"Missing from local  : {len(missing)}")
    log(f"Extra on local only : {len(extra)}")
    log("")

    if not missing:
        print("No missing subfolders. Local child has every blob parent subfolder.")
    else:
        print(f"{len(missing)} subfolder(s) on blob missing from local:\n")
        for name in missing:
            print(f"  {name}")

    if extra:
        print(f"\n{len(extra)} local-only subfolder(s) not on blob:\n")
        for name in extra:
            print(f"  {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
