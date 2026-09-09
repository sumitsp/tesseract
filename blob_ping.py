"""
Minimal Azure blob connectivity test.
Run this BEFORE file_counter / OCR scripts.

Usage:
  .\\venv\\Scripts\\python.exe blob_ping.py
"""
from __future__ import annotations

import sys
import traceback

from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient, ExponentialRetry

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"  # <-- set real container
PREFIX = "Run1/Batch1/DEID_PNGs/"


def main() -> None:
    if CONTAINER_NAME in ("", "YOUR_CONTAINER_NAME"):
        raise SystemExit("Set CONTAINER_NAME in blob_ping.py to your real container name.")

    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    print(f"Account : {STORAGE_ACCOUNT}")
    print(f"Container: {CONTAINER_NAME}")
    print(f"Prefix  : {PREFIX}")
    print()

    print("1) Getting token via AzureCliCredential (az login)...")
    cred = AzureCliCredential()
    token = cred.get_token("https://storage.azure.com/.default")
    print(f"   OK token expires: {token.expires_on}")

    print("2) Creating BlobServiceClient...")
    client = BlobServiceClient(
        account_url=account_url,
        credential=cred,
        retry_policy=ExponentialRetry(retry_total=5, initial_backoff=1),
        connection_timeout=60,
        read_timeout=120,
    )
    container = client.get_container_client(CONTAINER_NAME)

    print("3) get_container_properties()...")
    props = container.get_container_properties()
    print(f"   OK container={props.name}")

    print("4) list_blobs (first 5)...")
    count = 0
    for blob in container.list_blobs(name_starts_with=PREFIX):
        print(f"   - {blob.name}")
        count += 1
        if count >= 5:
            break
    if count == 0:
        print("   (no blobs under PREFIX — container/prefix may be wrong)")
    else:
        print(f"   OK listed {count} blob(s)")

    print()
    print("SUCCESS — blob access works from this Python/venv.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFAILED")
        traceback.print_exc()
        print(
            "\nAlso run this in PowerShell (same machine/VPN):\n"
            f"  az storage blob list --account-name {STORAGE_ACCOUNT} "
            f"--container-name {CONTAINER_NAME} --prefix \"{PREFIX}\" "
            "--auth-mode login --num-results 5 -o table\n"
        )
        sys.exit(1)
