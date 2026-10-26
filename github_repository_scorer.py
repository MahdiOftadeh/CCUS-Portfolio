#!/usr/bin/env python3
"""
github_repository_scorer.py - Unified Evidence Framework (Final)

Scores a Git repository 0-100 for a technical founder / elite freelancer by
combining automatic Git analysis with optional external evidence.

Weights:
    AUTOMATIC (Git-only, max 65):
        15  Delivery proxies (DORA-aligned: release tags, merges to main, CI)
        15  Engineering acceleration & commit discipline
        12  Production readiness (tests, error handling, CI)
        10  Verified AI leverage (claim vs code)
         8  Technical ownership & communication
         5  Activity continuity

    EXTERNAL (optional, max 35):
        15  Verified business impact (measurable KPI changes)
        10  Production delivery evidence (live URL / app store / deployment)
        10  Independent validation (client testimonials / references)

If `external_evidence.json` is missing, the automatic part is rescaled to
100 and a clear notice is printed.
"""

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, date
from pathlib import Path


# --------------------------------------------------------------------------
# Git helpers
# --------------------------------------------------------------------------
def _run_git(args, cwd):
    result = subprocess.run(
        ["git"] + args, cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError("git " + " ".join(args) + " failed:\n" + result.stderr.strip())
    return result.stdout


def _collect_commits(repo):
    out = _run_git(
        ["log", "--pretty=format:%H%x1f%an%x1f%ae%x1f%ad%x1f%s", "--date=short"],
        repo,
    )
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) != 5:
            continue
        sha, author, email, date_str, subject = parts
        commits.append({
            "sha": sha, "author": author, "email": email,
            "date": datetime.strptime(date_str, "%Y-%m-%d").date(),
            "subject": subject,
        })
    return commits


def _scan_repo_files(repo_root):
    paths = set()
    for p in repo_root.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            try:
                paths.add(str(p.relative_to(repo_root)).replace("\\", "/"))
            except ValueError:
                pass
    return paths


def _read_text_safely(path, max_chars=8000):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


# --------------------------------------------------------------------------
# AUTOMATIC COMPONENTS (max 65)
# --------------------------------------------------------------------------
def _score_delivery_proxies(repo_root, file_paths, commits):
    """15 pts: DORA-aligned proxies detectable from Git alone."""
    score = 0.0
    lower = {p.lower() for p in file_paths}

    tags = _run_git(["tag", "--list"], repo_root).splitlines()
    release_tags = [t for t in tags if re.match(r"v?\d+\.\d+", t)]
    if release_tags:
        score += 6

    merge_count = sum(1 for c in commits if "merge" in c["subject"].lower())
    if merge_count >= 3:
        score += 4

    if any(re.search(r"\.(github/workflows|gitlab-ci|circleci)", p) for p in lower):
        score += 5

    return min(15.0, score)


def _score_acceleration_discipline(commits):
    """15 pts: newer-half vs older-half commit rate + message quality."""
    if len(commits) < 5:
        return 0.0
    score = 0.0
    mid = len(commits) // 2
    older, newer = commits[mid:], commits[:mid]

    older_weeks = max(1, (older[0]["date"] - older[-1]["date"]).days / 7) if len(older) > 1 else 1
    newer_weeks = max(1, (newer[0]["date"] - newer[-1]["date"]).days / 7) if len(newer) > 1 else 1
    older_rate, newer_rate = len(older) / older_weeks, len(newer) / newer_weeks

    if older_rate > 0:
        ratio = newer_rate / older_rate
        if ratio >= 1.5:
            score += 6
        elif ratio >= 1.1:
            score += 4
        elif ratio >= 0.8:
            score += 2
        # decelerating gets 0

    trivial = re.compile(
        r"^(update|fix|test|merge|wip|temp|tmp|minor|typo|init|initial)\b",
        re.IGNORECASE,
    )
    meaningful = sum(
        1 for c in commits
        if len(c["subject"].strip()) >= 20 and not trivial.search(c["subject"])
    )
    ratio = meaningful / max(len(commits), 1)
    score += min(9.0, ratio * 9)
    return min(15.0, score)


def _score_production_readiness(repo_root, file_paths):
    """12 pts: tests, error handling, CI."""
    score = 0.0
    lower = {p.lower() for p in file_paths}
    if any(re.search(r"(^|/)(tests?|specs?)(/|$)", p) for p in lower):
        score += 5
    source = [p for p in file_paths if p.endswith((".py", ".js", ".ts", ".go"))]
    handled = sum(
        1 for rel in source[:50]
        if re.search(r"\b(try|except|catch|raise|throw)\b",
                     _read_text_safely(repo_root / rel, 4000), re.IGNORECASE)
    )
    if handled >= 2:
        score += 4
    if any(re.search(r"\.(github/workflows|gitlab-ci)", p) for p in lower):
        score += 3
    return min(12.0, score)


def _score_verified_ai(repo_root, file_paths):
    """10 pts: AI libs + LLM calls, bonus only if README claim matches code."""
    score = 0.0
    ai_libs = [r"\btorch\b", r"\btensorflow\b", r"\btransformers\b", r"\bopenai\b",
               r"\blangchain\b", r"\bllama_index\b", r"\banthropic\b", r"\bcohere\b",
               r"\bhuggingface\b", r"\bscikit-learn\b", r"\bxgboost\b"]
    llm_calls = [r"\.chat\.completions\.create\b", r"\.completions\.create\b",
                 r"Anthropic\(\)", r"OpenAI\(\)", r"pipeline\(.*task="]

    readme = next((p for p in file_paths if p.lower().endswith("readme.md")), None)
    claims_ai = bool(readme and re.search(
        r"\b(AI|LLM|machine learning|deep learning)\b",
        _read_text_safely(repo_root / readme, 8000), re.IGNORECASE))

    found_lib, found_call, scanned = False, False, 0
    for rel in file_paths:
        if scanned >= 150:
            break
        if not rel.endswith((".py", ".ipynb", ".ts", ".js", ".toml", ".txt")):
            continue
        txt = _read_text_safely(repo_root / rel, 6000)
        scanned += 1
        if not found_lib and any(re.search(p, txt, re.IGNORECASE) for p in ai_libs):
            found_lib = True
        if not found_call and any(re.search(p, txt) for p in llm_calls):
            found_call = True
        if found_lib and found_call:
            break

    if found_lib:
        score += 4
    if found_call:
        score += 3
    if claims_ai and (found_lib or found_call):
        score += 3
    return min(10.0, score)


def _score_ownership_communication(repo_root, file_paths, commits):
    """8 pts: architecture docs, refactor commits, LICENSE."""
    score = 0.0
    lower = {p.lower() for p in file_paths}
    if any("architect" in p for p in lower):
        score += 3
    if sum(1 for c in commits if "refactor" in c["subject"].lower()) >= 3:
        score += 3
    if any(p.lower().startswith(("license", "licence")) for p in file_paths):
        score += 2
    return min(8.0, score)


def _score_activity_continuity(commits):
    """5 pts: active-day ratio over the commit span."""
    if len(commits) < 2:
        return 0.0
    dates = sorted({c["date"] for c in commits})
    span = max(1, (dates[-1] - dates[0]).days + 1)
    return min(5.0, len(dates) / span * 5)


# --------------------------------------------------------------------------
# EXTERNAL EVIDENCE (max 35, optional)
# --------------------------------------------------------------------------
def _load_external_evidence(repo_root):
    evidence_path = repo_root / "external_evidence.json"
    if not evidence_path.exists():
        return None
    try:
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Warning: external_evidence.json could not be parsed: " + str(exc),
              file=sys.stderr)
        return None


def _score_business_impact(evidence):
    """15 pts: verified KPI changes (baseline -> result)."""
    impacts = evidence.get("business_impact", [])
    if not impacts:
        return 0.0
    score = 0.0
    for item in impacts:
        if not isinstance(item, dict):
            continue
        # Require baseline, result, and a verifiable source reference
        has_baseline = item.get("baseline") not in (None, "", "N/A")
        has_result = item.get("result") not in (None, "", "N/A")
        has_source = bool(item.get("source", "").strip())
        if has_baseline and has_result:
            score += 3
            if has_source:
                score += 2  # bonus for verifiability
    return min(15.0, score)


def _score_delivery_evidence(evidence):
    """10 pts: live URL, app store link, deployment artifacts."""
    score = 0.0
    if evidence.get("live_url", "").strip():
        score += 5
    if evidence.get("app_store_url", "").strip():
        score += 3
    if evidence.get("deployment_artifacts"):
        score += 2
    return min(10.0, score)


def _score_independent_validation(evidence):
    """10 pts: client testimonials / references with names or companies."""
    validations = evidence.get("client_validations", [])
    if not validations:
        return 0.0
    score = 0.0
    for v in validations:
        if not isinstance(v, dict):
            continue
        # A validation counts only if it names a client (person or company)
        # and includes a verifiable quote or reference
        has_client = bool(str(v.get("client", "")).strip())
        has_quote = len(str(v.get("quote", "")).strip()) >= 20
        has_reference = bool(str(v.get("reference", "")).strip())
        if has_client and (has_quote or has_reference):
            score += 3
            if has_quote and has_reference:
                score += 1  # extra for both quote and contactable reference
    return min(10.0, score)


def _compute_external_score(evidence):
    if evidence is None:
        return 0.0, {}, False
    b = _score_business_impact(evidence)
    d = _score_delivery_evidence(evidence)
    v = _score_independent_validation(evidence)
    total = b + d + v
    breakdown = {
        "ext_business_impact (15)": round(b, 1),
        "ext_delivery_evidence (10)": round(d, 1),
        "ext_independent_validation (10)": round(v, 1),
    }
    return round(total, 1), breakdown, True


# --------------------------------------------------------------------------
# Core scoring
# --------------------------------------------------------------------------
def compute_score(repo_path):
    repo = Path(repo_path).resolve()
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise RuntimeError(str(repo) + " is not a Git repository.")

    commits = _collect_commits(repo)
    if not commits:
        return 0.0, {}, False

    file_paths = _scan_repo_files(repo)

    auto = {
        "delivery_proxies (15)": _score_delivery_proxies(repo, file_paths, commits),
        "acceleration_discipline (15)": _score_acceleration_discipline(commits),
        "production_readiness (12)": _score_production_readiness(repo, file_paths),
        "verified_ai (10)": _score_verified_ai(repo, file_paths),
        "ownership_communication (8)": _score_ownership_communication(
            repo, file_paths, commits),
        "activity_continuity (5)": _score_activity_continuity(commits),
    }
    auto_total = sum(auto.values())

    evidence = _load_external_evidence(repo)
    ext_total, ext_breakdown, external_present = _compute_external_score(evidence)

    if external_present:
        # Automatic (65) + External (35)
        final = auto_total * (100.0 / 65.0) * 0.65 + ext_total * (100.0 / 35.0) * 0.35
    else:
        # Rescale automatic 65 -> 100
        final = auto_total * (100.0 / 65.0)

    final = round(max(0.0, min(100.0, final)), 1)

    breakdown = {}
    breakdown.update({k: round(v, 1) for k, v in auto.items()})
    breakdown["auto_subtotal (65)"] = round(auto_total, 1)
    if external_present:
        breakdown.update(ext_breakdown)
        breakdown["ext_subtotal (35)"] = ext_total
    breakdown["external_evidence_present"] = external_present

    return final, breakdown, external_present


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="github_repository_scorer.py",
        description="Unified Evidence Framework: Git analysis + optional external evidence.",
    )
    parser.add_argument("repo", nargs="?", default=".", help="Path to Git repo.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print breakdown before final score.")
    args = parser.parse_args()

    try:
        score, breakdown, external_present = compute_score(args.repo)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr)
        sys.exit(1)

    if args.verbose and breakdown:
        print("Unified Evidence Breakdown:")
        for k, v in breakdown.items():
            print("  " + str(k) + ": " + str(v))
        print()

    if not external_present:
        print("Notice: external_evidence.json not found. Score reflects Git-only "
              "signals (rescaled to 100). Add external evidence to unlock the "
              "full 35-point external layer.")
        print()

    print("Final Score: " + str(score) + " / 100")


if __name__ == "__main__":
    main()