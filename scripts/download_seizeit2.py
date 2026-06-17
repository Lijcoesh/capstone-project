#!/usr/bin/env python3
"""
Download SeizeIT2 (OpenNeuro ds005873 v1.1.0) from the public S3 bucket.

No AWS account or AWS CLI install is required — this uses anonymous S3 access
via boto3 (same data as ``aws s3 sync --no-sign-request s3://openneuro.org/ds005873``).

By default, downloads EEG and ECG files for subjects sub-001 through sub-060 into
data/raw/seizeit2/, matching this project's BIDS layout. Existing files with the
correct size are skipped so you can resume an interrupted download.

Setup:
    pip install boto3 tqdm

Examples:
    python scripts/download_seizeit2.py
    python scripts/download_seizeit2.py --first 1 --last 60
    python scripts/download_seizeit2.py --subjects 1,5,10-15
    python scripts/download_seizeit2.py --modalities eeg,ecg,emg,mov
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import boto3
    import urllib3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError, SSLError
except ImportError:
    print("Missing dependency. Install with: pip install boto3 tqdm", file=sys.stderr)
    sys.exit(1)

from tqdm import tqdm

BUCKET = "openneuro.org"
DATASET = "ds005873"
DATASET_PREFIX = f"{DATASET}/"
DEFAULT_MODALITIES = ("eeg", "ecg")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "seizeit2"


def parse_subject_list(spec: str) -> list[int]:
    """Parse '1,3,10-15' into sorted unique subject numbers."""
    numbers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return sorted(numbers)


def subject_dir(subject_num: int) -> str:
    return f"sub-{subject_num:03d}"


def make_s3_client(*, verify):
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        verify=verify,
    )


def connect_s3(*, no_verify_ssl: bool):
    """Connect to public OpenNeuro S3, auto-handling broken local CA stores."""
    if no_verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return make_s3_client(verify=False)

    try:
        import certifi

        verify: bool | str = certifi.where()
    except ImportError:
        verify = True

    s3 = make_s3_client(verify=verify)
    try:
        s3.list_objects_v2(Bucket=BUCKET, Prefix=DATASET_PREFIX, MaxKeys=1)
        return s3
    except SSLError:
        print(
            "Warning: SSL certificate verification failed on this machine; "
            "retrying without verification.",
            file=sys.stderr,
        )
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return make_s3_client(verify=False)


def list_keys(s3, prefix: str) -> list[tuple[str, int]]:
    """Return (s3_key, size_bytes) for every object under prefix."""
    keys: list[tuple[str, int]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            keys.append((key, int(obj["Size"])))
    return keys


def local_path_for_key(out_dir: Path, key: str) -> Path:
    if not key.startswith(DATASET_PREFIX):
        raise ValueError(f"Unexpected S3 key (expected {DATASET_PREFIX}…): {key}")
    return out_dir / key[len(DATASET_PREFIX) :]


def should_skip(path: Path, expected_size: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_size


def download_one(s3, key: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, key, str(dest))
    return key


def key_matches_modalities(key: str, modalities: set[str]) -> bool:
    if not modalities:
        return True
    # ds005873/sub-001/ses-01/eeg/...
    parts = key.split("/")
    if len(parts) < 4:
        return True
    return parts[3] in modalities


def collect_downloads(
    s3,
    out_dir: Path,
    prefixes: list[str],
    *,
    modalities: set[str],
) -> tuple[list[tuple[str, Path, int]], int]:
    """Gather files to download. Returns (jobs, skipped_count)."""
    jobs: list[tuple[str, Path, int]] = []
    skipped = 0
    seen: set[str] = set()

    for prefix in prefixes:
        for key, size in list_keys(s3, prefix):
            if key in seen:
                continue
            if not key_matches_modalities(key, modalities):
                continue
            seen.add(key)
            dest = local_path_for_key(out_dir, key)
            if should_skip(dest, size):
                skipped += 1
                continue
            jobs.append((key, dest, size))

    return jobs, skipped


def run_downloads(
    s3,
    jobs: list[tuple[str, Path, int]],
    *,
    workers: int,
) -> None:
    if not jobs:
        return

    total_bytes = sum(size for _, _, size in jobs)
    bar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading",
    )

    def _task(item: tuple[str, Path, int]) -> tuple[str, int]:
        key, dest, size = item
        download_one(s3, key, dest)
        return key, size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_task, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                _, size = fut.result()
                bar.update(size)
            except (ClientError, SSLError) as exc:
                bar.close()
                raise SystemExit(f"S3 download failed: {exc}") from exc

    bar.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download SeizeIT2 (OpenNeuro ds005873) subjects from public S3.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--first",
        type=int,
        default=1,
        help="First subject number when using the default range (default: 1)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=60,
        help="Last subject number when using the default range (default: 60)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Explicit subject list, e.g. '1,5,10-15' (overrides --first/--last)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=",".join(DEFAULT_MODALITIES),
        help="Comma-separated BIDS modality folders to fetch "
        f"(default: {','.join(DEFAULT_MODALITIES)})",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Also download dataset-level BIDS files (dataset_description.json, etc.)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without fetching them",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification (use only if downloads fail with cert errors)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.subjects:
        subject_nums = parse_subject_list(args.subjects)
    else:
        if args.first < 1 or args.last < args.first:
            raise SystemExit("--first must be >= 1 and --last must be >= --first")
        subject_nums = list(range(args.first, args.last + 1))

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [f"{DATASET_PREFIX}{subject_dir(n)}/" for n in subject_nums]
    if args.include_root:
        prefixes.insert(0, DATASET_PREFIX)

    print(
        f"OpenNeuro {DATASET}: subjects "
        f"{subject_nums[0]:03d}-{subject_nums[-1]:03d} "
        f"({len(subject_nums)} total) -> {out_dir}"
    )

    modalities = {
        m.strip().lower()
        for m in args.modalities.split(",")
        if m.strip()
    }
    print(f"Modalities: {', '.join(sorted(modalities))}")

    s3 = connect_s3(no_verify_ssl=args.no_verify_ssl)

    try:
        jobs, skipped = collect_downloads(s3, out_dir, prefixes, modalities=modalities)
    except (ClientError, SSLError) as exc:
        code = ""
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
        raise SystemExit(f"S3 error ({code}): {exc}") from exc

    edf_jobs = [j for j in jobs if j[0].endswith(".edf")]
    print(
        f"Found {len(jobs)} file(s) to download "
        f"({len(edf_jobs)} .edf signal file(s)); skipped {skipped} already complete."
    )

    if args.dry_run:
        for key, dest, size in sorted(jobs, key=lambda x: x[0]):
            print(f"  {size:>12,d}  {dest}")
        return

    if not jobs:
        print("Nothing to download.")
        return

    run_downloads(s3, jobs, workers=max(1, args.workers))
    print("Done.")


if __name__ == "__main__":
    main()
