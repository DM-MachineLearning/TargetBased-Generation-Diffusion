
"""Download PLINDER datasets from the public GCS bucket.

Usage examples:
  python scripts/download_plinder.py --release 2024-06 --iteration v2 --out data/plinder --prefix splits
  python scripts/download_plinder.py --release 2024-06 --iteration v2 --out data/plinder --prefix systems/1a
  python scripts/download_plinder.py --release 2024-06 --iteration v2 --out data/plinder --prefix systems --max_files 200

Notes:
- Uses anonymous GCS access (public bucket) via gcsfs.
- For large downloads, gsutil is faster; this is a pure-Python alternative.
"""

import argparse, os
from pathlib import Path
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="2024-06")
    ap.add_argument("--iteration", default="v2")
    ap.add_argument("--out", default="data/plinder")
    ap.add_argument("--prefix", default="splits")
    ap.add_argument("--max_files", type=int, default=0, help="If >0, limit number of files downloaded (debug).")
    args = ap.parse_args()

    try:
        import gcsfs
    except Exception as e:
        raise SystemExit("Missing dependency gcsfs. Install via pip or conda env. Error: %r" % (e,))

    fs = gcsfs.GCSFileSystem(token="anon")
    gcs_prefix = f"plinder/{args.release}/{args.iteration}/{args.prefix}".rstrip("/")
    out_dir = Path(args.out) / args.release / args.iteration
    out_dir.mkdir(parents=True, exist_ok=True)

    objs = fs.find(gcs_prefix)
    if not objs:
        raise SystemExit(f"No objects found under gs://{gcs_prefix} (check release/iteration/prefix).")

    if args.max_files and args.max_files > 0:
        objs = objs[: args.max_files]

    for obj in tqdm(objs, desc=f"Downloading {args.prefix}", unit="file"):
        rel = obj.split(f"plinder/{args.release}/{args.iteration}/", 1)[1]
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Skip if exists
        if dst.exists() and dst.stat().st_size > 0:
            continue
        fs.get(obj, str(dst))

    print("Done. Data at:", out_dir)

if __name__ == "__main__":
    main()
