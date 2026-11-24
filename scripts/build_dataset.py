"""Download openly licensed primate photos from iNaturalist into data/<class>/.

    python scripts/build_dataset.py --per-class 100

Only research-grade observations with a Creative Commons licence are used, and
only the FIRST photo of each observation, so no two photos share an observation
(this keeps the train/validation/test split free of same-animal leakage).
Credits go to data/LICENSES.csv. Photos are not committed (see .gitignore).
"""
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.inaturalist.org/v1/observations"
TAXA = {"chimpanzee": "Pan troglodytes", "gorilla": "Gorilla", "orangutan": "Pongo"}
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"  # no -nd: we resize photos


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "primates-classifier-dataset-builder"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-class", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    rows = []
    for cls, taxon in TAXA.items():
        (out / cls).mkdir(parents=True, exist_ok=True)
        got, page = 0, 1
        while got < args.per_class:
            q = urllib.parse.urlencode({
                "taxon_name": taxon, "quality_grade": "research", "photo_license": LICENSES,
                "photos": "true", "per_page": 50, "page": page, "order_by": "id"})
            results = json.loads(get(f"{API}?{q}"))["results"]
            if not results:
                break
            for obs in results:
                photo = obs["photos"][0]
                lic = photo.get("license_code")
                if not lic or got >= args.per_class:
                    continue
                name = f"{obs['id']}.jpg"
                (out / cls / name).write_bytes(get(photo["url"].replace("square", "medium")))
                rows.append([cls, name, obs["id"], lic, (obs.get("user") or {}).get("login", ""), obs.get("uri", "")])
                got += 1
                time.sleep(0.2)  # be polite to the API
            page += 1
        print(f"{cls}: {got} photos")
    with open(out / "LICENSES.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "observation_id", "license", "observer", "url"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
