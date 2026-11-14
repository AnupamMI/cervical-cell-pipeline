#!/usr/bin/env python3
"""Minimal GitHub REST helper for releases. The token is read ONLY via `git credential fill` inside this process
and is never printed, logged or written anywhere.

    python gh_api.py release  --tag v0.1-models --name "..." --body "..." [--target main] [--asset FILE ...]
    python gh_api.py get-release --tag v0.1-models
"""
import argparse, json, mimetypes, subprocess, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

REPO = "AnupamMI/cervical-cell-pipeline"


def _token():
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True, timeout=30).stdout
    tok = dict(l.split("=", 1) for l in out.splitlines() if "=" in l).get("password")
    if not tok:
        sys.exit("no GitHub credential available from git's credential helper")
    return tok


def _call(method, url, token, data=None, ctype="application/json", timeout=60):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                          "X-GitHub-Api-Version": "2022-11-28", "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e)
        except Exception:
            msg = {}
        return e.code, msg


def get_release(tag, token):
    return _call("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("release", "get-release"))
    ap.add_argument("--tag", required=True); ap.add_argument("--name", default=None); ap.add_argument("--body", default="")
    ap.add_argument("--target", default="main"); ap.add_argument("--asset", nargs="*", default=[])
    a = ap.parse_args()
    tok = _token()
    code, rel = get_release(a.tag, tok)
    if a.cmd == "get-release":
        print(json.dumps({"status": code, "url": rel.get("html_url"), "assets": [(x["name"], x["size"]) for x in rel.get("assets", [])]}))
        return
    if code == 404:
        code, rel = _call("POST", f"https://api.github.com/repos/{REPO}/releases", tok,
                          {"tag_name": a.tag, "name": a.name or a.tag, "body": a.body, "target_commitish": a.target,
                           "draft": False, "prerelease": False})
        if code not in (200, 201):
            sys.exit(f"create release failed: HTTP {code} {rel.get('message')}")
        print(f"release created: {rel['html_url']}")
    elif code == 200:
        print(f"release exists: {rel['html_url']}")
    else:
        sys.exit(f"get release failed: HTTP {code} {rel.get('message')}")
    have = {x["name"]: x["size"] for x in rel.get("assets", [])}
    upload = rel["upload_url"].split("{")[0]
    for f in map(Path, a.asset):
        if f.name in have and have[f.name] == f.stat().st_size:
            print(f"asset {f.name}: already uploaded ({have[f.name]} bytes)"); continue
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        code, r = _call("POST", f"{upload}?name={urllib.parse.quote(f.name)}", tok, f.read_bytes(), ctype, timeout=1800)
        if code not in (200, 201):
            sys.exit(f"upload {f.name} failed: HTTP {code} {r.get('message')}")
        print(f"asset {f.name}: uploaded ({r['size']} bytes) {r['browser_download_url']}")


if __name__ == "__main__":
    main()
