#!/usr/bin/env python3
"""
github_repository_scorer.py - Automatic Git Activity Scorer (100-point)

Scores a Git repository from 0 to 100 using only signals detectable from
Git and the repository content. Designed for a technical founder or
freelancer building a portfolio before acquiring clients.

No external_evidence.json is used. The score is honest and fully
reproducible from the repository alone.

Components (sum = 100):
    25  Delivery Evidence          (release tags, merge discipline, CI workflows)
    20  Production Readiness       (tests, error handling, CI)
    20  Acceleration & Discipline  (commit rate trend + message quality)
    15  Verified AI Leverage       (AI libraries + LLM calls, claim vs code)
    12  Ownership & Communication  (architecture docs, refactor commits, LICENSE)
     8  Activity Continuity        (active-day ratio over the commit span)

Usage:
    python github_repository_scorer.py
    python github_repository_scorer.py --verbose
    python github_repository_scorer.py --verbose --report report.md
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
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
# Component 1: Delivery Evidence (max 25)
# --------------------------------------------------------------------------
def _score_delivery(repo_root, file_paths, commits):
    """Release tags (10) + merge discipline (7) + CI workflows (8)."""
    score = 0.0
    lower = {p.lower() for p in file_paths}

    tags = _run_git(["tag", "--list"], repo_root).splitlines()
    release_tags = [t for t in tags if re.match(r"v?\d+\.\d+", t)]
    if release_tags:
        score += 10

    merge_count = sum(1 for c in commits if "merge" in c["subject"].lower())
    if merge_count >= 3:
        score += 7

    has_ci = any(re.search(r"\.(github/workflows|gitlab-ci|circleci)", p) for p in lower)
    if has_ci:
        score += 8

    return min(25.0, score), {
        "release_tags_present": bool(release_tags),
        "release_tag_count": len(release_tags),
        "merge_commits": merge_count,
        "ci_workflows": has_ci,
    }


# --------------------------------------------------------------------------
# Component 2: Production Readiness (max 20)
# --------------------------------------------------------------------------
def _score_production(repo_root, file_paths):
    """Tests (8) + error handling (7) + CI (5)."""
    score = 0.0
    lower = {p.lower() for p in file_paths}

    has_tests = any(re.search(r"(^|/)(tests?|specs?)(/|$)", p) for p in lower)
    if has_tests:
        score += 8

    source = [p for p in file_paths if p.endswith((".py", ".js", ".ts", ".go"))]
    handled = sum(
        1 for rel in source[:50]
        if re.search(r"\b(try|except|catch|raise|throw)\b",
                     _read_text_safely(repo_root / rel, 4000), re.IGNORECASE)
    )
    if handled >= 2:
        score += 7

    has_ci = any(re.search(r"\.(github/workflows|gitlab-ci)", p) for p in lower)
    if has_ci:
        score += 5

    return min(20.0, score), {
        "tests_present": has_tests,
        "files_with_error_handling": handled,
        "ci_present": has_ci,
    }


# --------------------------------------------------------------------------
# Component 3: Acceleration & Discipline (max 20)
# --------------------------------------------------------------------------
def _score_acceleration(commits):
    """Commit rate trend (8) + meaningful message ratio (12)."""
    if len(commits) < 5:
        return 0.0, {"reason": "fewer than 5 commits"}

    score = 0.0
    mid = len(commits) // 2
    older, newer = commits[mid:], commits[:mid]

    older_weeks = max(1, (older[0]["date"] - older[-1]["date"]).days / 7) if len(older) > 1 else 1
    newer_weeks = max(1, (newer[0]["date"] - newer[-1]["date"]).days / 7) if len(newer) > 1 else 1
    older_rate, newer_rate = len(older) / older_weeks, len(newer) / newer_weeks

    ratio = 0.0
    if older_rate > 0:
        ratio = newer_rate / older_rate
        if ratio >= 1.5:
            score += 8
        elif ratio >= 1.1:
            score += 5
        elif ratio >= 0.8:
            score += 3

    trivial = re.compile(
        r"^(update|fix|test|merge|wip|temp|tmp|minor|typo|init|initial)\b",
        re.IGNORECASE,
    )
    meaningful = sum(
        1 for c in commits
        if len(c["subject"].strip()) >= 20 and not trivial.search(c["subject"])
    )
    meaningful_ratio = meaningful / max(len(commits), 1)
    score += meaningful_ratio * 12

    return min(20.0, score), {
        "older_rate_per_week": round(older_rate, 2),
        "newer_rate_per_week": round(newer_rate, 2),
        "acceleration_ratio": round(ratio, 2),
        "meaningful_commits": meaningful,
        "meaningful_ratio": round(meaningful_ratio, 3),
    }


# --------------------------------------------------------------------------
# Component 4: Verified AI (max 15)
# --------------------------------------------------------------------------
def _score_verified_ai(repo_root, file_paths):
    """AI libs (6) + LLM calls (5) + claim-code match bonus (4)."""
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
        score += 6
    if found_call:
        score += 5
    if claims_ai and (found_lib or found_call):
        score += 4

    return min(15.0, score), {
        "readme_claims_ai": claims_ai,
        "ai_library_found": found_lib,
        "llm_call_found": found_call,
    }


# --------------------------------------------------------------------------
# Component 5: Ownership & Communication (max 12)
# --------------------------------------------------------------------------
def _score_ownership(repo_root, file_paths, commits):
    """Architecture docs (5) + refactor commits (4) + LICENSE (3)."""
    score = 0.0
    lower = {p.lower() for p in file_paths}

    has_arch = any("architect" in p for p in lower)
    if has_arch:
        score += 5

    refactor_count = sum(1 for c in commits if "refactor" in c["subject"].lower())
    if refactor_count >= 3:
        score += 4

    has_license = any(p.lower().startswith(("license", "licence")) for p in file_paths)
    if has_license:
        score += 3

    return min(12.0, score), {
        "architecture_doc_present": has_arch,
        "refactor_commits": refactor_count,
        "license_present": has_license,
    }


# --------------------------------------------------------------------------
# Component 6: Activity Continuity (max 8)
# --------------------------------------------------------------------------
def _score_continuity(commits):
    """Active-day ratio over the commit span."""
    if len(commits) < 2:
        return 0.0, {"reason": "fewer than 2 commits"}
    dates = sorted({c["date"] for c in commits})
    span = max(1, (dates[-1] - dates[0]).days + 1)
    active_days = len(dates)
    ratio = active_days / span
    return min(8.0, ratio * 8), {
        "active_days": active_days,
        "span_days": span,
        "active_ratio": round(ratio, 3),
    }


# --------------------------------------------------------------------------
# Core scoring
# --------------------------------------------------------------------------
def compute_score(repo_path):
    repo = Path(repo_path).resolve()
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise RuntimeError(str(repo) + " is not a Git repository.")

    commits = _collect_commits(repo)
    if not commits:
        return 0.0, {}, {}

    file_paths = _scan_repo_files(repo)

    d_score, d_details = _score_delivery(repo, file_paths, commits)
    p_score, p_details = _score_production(repo, file_paths)
    a_score, a_details = _score_acceleration(commits)
    ai_score, ai_details = _score_verified_ai(repo, file_paths)
    o_score, o_details = _score_ownership(repo, file_paths, commits)
    c_score, c_details = _score_continuity(commits)

    total = round(d_score + p_score + a_score + ai_score + o_score + c_score, 1)
    total = max(0.0, min(100.0, total))

    breakdown = {
        "Delivery Evidence (25)": (round(d_score, 1), d_details),
        "Production Readiness (20)": (round(p_score, 1), p_details),
        "Acceleration & Discipline (20)": (round(a_score, 1), a_details),
        "Verified AI (15)": (round(ai_score, 1), ai_details),
        "Ownership & Communication (12)": (round(o_score, 1), o_details),
        "Activity Continuity (8)": (round(c_score, 1), c_details),
    }
    return total, breakdown, {"commits": len(commits), "files": len(file_paths)}


# --------------------------------------------------------------------------
# Suggestions engine
# --------------------------------------------------------------------------
def _generate_suggestions(breakdown):
    suggestions = []

    _, d = breakdown["Delivery Evidence (25)"]
    if not d.get("release_tags_present"):
        suggestions.append(
            "Create a release tag: `git tag -a v0.1.0 -m 'First release' && git push --tags` (+10)"
        )
    if d.get("merge_commits", 0) < 3:
        suggestions.append(
            "Use feature branches and merge them to main (+7 when >= 3 merges)"
        )
    if not d.get("ci_workflows"):
        suggestions.append("Add a CI workflow (.github/workflows/test.yml) (+8)")

    _, p = breakdown["Production Readiness (20)"]
    if not p.get("tests_present"):
        suggestions.append("Add a `tests/` directory with real tests (+8)")
    if p.get("files_with_error_handling", 0) < 2:
        suggestions.append(
            "Add try/except or raise statements to at least 2 source files (+7)"
        )
    if not p.get("ci_present"):
        suggestions.append("Set up CI to run tests automatically (+5)")

    _, a = breakdown["Acceleration & Discipline (20)"]
    if a.get("meaningful_ratio", 1) < 0.8:
        suggestions.append(
            "Write descriptive commit messages (>= 20 chars, avoid 'update'/'fix' alone) (+12)"
        )

    _, ai = breakdown["Verified AI (15)"]
    if not ai.get("ai_library_found"):
        suggestions.append(
            "If you use AI/ML, add the library (scikit-learn, torch, openai) to code (+6)"
        )
    if not ai.get("llm_call_found"):
        suggestions.append(
            "If you call an LLM, expose it in code (e.g., openai.chat.completions.create) (+5)"
        )

    _, o = breakdown["Ownership & Communication (12)"]
    if not o.get("architecture_doc_present"):
        suggestions.append("Write an `ARCHITECTURE.md` describing module layout (+5)")
    if o.get("refactor_commits", 0) < 3:
        suggestions.append("Use `refactor:` commit prefix for refactoring work (+4)")
    if not o.get("license_present"):
        suggestions.append("Add a LICENSE file (MIT, Apache-2.0, etc.) (+3)")

    return suggestions


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def _write_report(score, breakdown, meta, repo_path, report_path):
    lines = []
    lines.append("# Git Repository Activity Report")
    lines.append("")
    lines.append("**Repository:** `" + Path(repo_path).resolve().name + "`  ")
    lines.append("**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  ")
    lines.append("**Method:** 100-point automatic scoring (Git-only, no external evidence)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Score Breakdown")
    lines.append("")
    lines.append("| Component | Score | Max |")
    lines.append("|---|---:|---:|")
    for name, (val, _details) in breakdown.items():
        max_v = re.search(r"\((\d+)\)", name).group(1)
        clean_name = re.sub(r"\s*\(\d+\)", "", name)
        lines.append("| " + clean_name + " | " + str(val) + " | " + max_v + " |")
    lines.append("| **Final** | **" + str(score) + "** | **100** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final Score")
    lines.append("")
    lines.append("### **" + str(score) + " / 100**")
    lines.append("")

    if score >= 80:
        interp = "Excellent — strong delivery, discipline, and impact signals."
    elif score >= 60:
        interp = "Good — solid foundation with visible room for growth."
    elif score >= 40:
        interp = "Fair — real activity, but gaps in delivery, discipline, or verification."
    elif score >= 20:
        interp = "Weak — limited verifiable evidence of delivery or quality."
    else:
        interp = "Very low — minimal verifiable evidence."
    lines.append("**Interpretation:** " + interp)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Signals")
    lines.append("")
    for name, (_val, details) in breakdown.items():
        lines.append("### " + re.sub(r"\s*\(\d+\)", "", name))
        lines.append("")
        if not details:
            lines.append("_No signals collected._")
        else:
            for k, v in details.items():
                lines.append("- **" + k + ":** " + str(v))
        lines.append("")

    suggestions = _generate_suggestions(breakdown)
    lines.append("---")
    lines.append("")
    lines.append("## How to Improve Your Score")
    lines.append("")
    if suggestions:
        for s in suggestions:
            lines.append("- " + s)
    else:
        lines.append("All components are at maximum. Focus on shipping real releases and seeking client validation.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by github_repository_scorer.py (100-point, Git-only).*")

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="github_repository_scorer.py",
        description="Automatic 100-point Git activity score. No external evidence.",
    )
    parser.add_argument("repo", nargs="?", default=".", help="Path to Git repo.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed breakdown before final score.")
    parser.add_argument("--report", "-r", default=None,
                        help="Write a detailed Markdown report to this path.")
    args = parser.parse_args()

    try:
        score, breakdown, meta = compute_score(args.repo)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr)
        sys.exit(1)

    if args.verbose and breakdown:
        print("Score Breakdown (100-point, Git-only):")
        print("  Commits: " + str(meta.get("commits", 0)))
        print("  Files:   " + str(meta.get("files", 0)))
        print("")
        for name, (val, _details) in breakdown.items():
            max_v = re.search(r"\((\d+)\)", name).group(1)
            clean_name = re.sub(r"\s*\(\d+\)", "", name)
            print("  " + clean_name + ": " + str(val) + " / " + max_v)
        print("")

    if args.report:
        _write_report(score, breakdown, meta, args.repo, args.report)
        print("Report written to " + args.report)
        print("")

    print("Final Score: " + str(score) + " / 100")


if __name__ == "__main__":
    main()