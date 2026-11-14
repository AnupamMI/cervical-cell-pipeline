#!/usr/bin/env python3
"""Download weights that are too large for git (GitHub Release assets) and verify their SHA256.

    GH_TOKEN=... python scripts/fetch_release_weights.py [--tag v0.1-models]

The token is read from the GH_TOKEN environment variable (needed because the repository is private) and is never
printed or written anywhere. Files already present with the right SHA256 are skipped.
"""
import argparse, hashlib, json, os, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "AnupamMI/cervical-cell-pipeline"
# asset name -> (destination relative to the repo, sha256)
ASSETS = {
    "yolo26l_1280_best.pt": ("models/detector_b/yolo26l_1280_best.pt",
                             "703f3e716a5116b98d581497eed6bab62b49181577aeb2062f209cbb230fb72b"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def api(url, token, accept="application/vnd.github+json"):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": accept,
                                               "X-GitHub-Api-Version": "2022-11-28"})
    return urllib.request.urlopen(req, timeout=120)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def download(asset_api_url, token):
    """Asset API URL -> 302 to a signed storage URL, which must be fetched WITHOUT the Authorization header."""
    req = urllib.request.Request(asset_api_url, headers={"Authorization": f"Bearer {token}",
                                                         "Accept": "application/octet-stream"})
    try:
        return urllib.request.build_opener(_NoRedirect).open(req, timeout=120)   # served directly (no redirect)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        return urllib.request.urlopen(e.headers["Location"], timeout=600)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v0.1-models")
    a = ap.parse_args()
    token = os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("GH_TOKEN is not set (the repository is private)")
    try:
        with api(f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}", token) as r:
            rel = json.load(r)
    except Exception as e:
        sys.exit(f"cannot read release {a.tag}: {type(e).__name__} {getattr(e, 'code', '')}")
    assets = {x["name"]: x for x in rel.get("assets", [])}
    ok = True
    for name, (dest, want) in ASSETS.items():
        dest = ROOT / dest
        if dest.exists() and sha256(dest) == want:
            print(f"{name}: already present, SHA256 ok"); continue
        if name not in assets:
            print(f"{name}: not in release {a.tag}"); ok = False; continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with download(assets[name]["url"], token) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        got = sha256(tmp)
        if got != want:
            tmp.unlink(); print(f"{name}: SHA256 mismatch ({got[:12]}... != {want[:12]}...), not installed"); ok = False; continue
        tmp.rename(dest)
        print(f"{name}: downloaded to {dest.relative_to(ROOT)}, SHA256 ok")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
