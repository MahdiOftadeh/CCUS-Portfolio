import os, sys, subprocess, json, glob
from pathlib import Path

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

print("[*] Auditing Git History & Remote...")
origin_url = run_cmd("git config --get remote.origin.url")
commit_count = run_cmd("git rev-list --count HEAD")
last_commit = run_cmd('git log -1 --format="%cd | %s (%h)"')
branches = run_cmd("git branch -a")

print("[*] Analyzing Directory Hygiene...")
total_files = 0
untracked_junk = []
large_files = []
py_files = []
total_py_lines = 0
docstring_count = 0
type_hint_count = 0

for root, dirs, files in os.walk('.'):
    if '.git' in root:
        continue
    for f in files:
        fp = os.path.join(root, f)
        total_files += 1
        try:
            sz = os.path.getsize(fp)
            if sz > 5 * 1024 * 1024:  # > 5MB
                large_files.append((fp, round(sz / (1024*1024), 2)))
        except OSError:
            pass

        if f.endswith(('.tmp', '.pyc', '.bak', '.txt')) and ('(' in f or 'copy' in f.lower()):
            untracked_junk.append(fp)

        if f.endswith('.py'):
            py_files.append(fp)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as pfile:
                    for line in pfile:
                        total_py_lines += 1
                        sline = line.strip()
                        if sline.startswith(('"""', "'''")):
                            docstring_count += 1
                        if ' -> ' in sline or ': float' in sline or ': int' in sline or ': str' in sline:
                            type_hint_count += 1
            except Exception:
                pass

print("[*] Checking CI/CD, Tests & Packaging...")
has_readme = any(os.path.exists(f) for f in ['README.md', 'README.rst', 'README.txt'])
has_license = any(os.path.exists(f) for f in ['LICENSE', 'LICENSE.md', 'LICENSE.txt'])
has_pyproject = os.path.exists('pyproject.toml') or os.path.exists('setup.py')
has_requirements = os.path.exists('requirements.txt') or os.path.exists('environment.yml')
has_workflows = os.path.exists('.github/workflows')

test_files = [f for f in py_files if 'test' in os.path.basename(f).lower()]
pytest_output = run_cmd('pytest --collect-only -q') if test_files else 'No test runner executed'

report = {
    'remote_origin': origin_url,
    'total_commits': commit_count,
    'last_commit': last_commit,
    'total_files_in_tree': total_files,
    'python_files_count': len(py_files),
    'total_python_lines': total_py_lines,
    'docstring_markers': docstring_count,
    'type_hint_markers': type_hint_count,
    'large_files_above_5MB': large_files,
    'duplicate_or_temp_candidates': untracked_junk[:20],
    'total_duplicate_candidates': len(untracked_junk),
    'standards': {
        'has_readme': has_readme,
        'has_license': has_license,
        'has_pyproject_or_setup': has_pyproject,
        'has_requirements': has_requirements,
        'has_github_actions': has_workflows,
    },
    'tests': {
        'test_files_detected': test_files,
        'pytest_dry_run': pytest_output[:300] if pytest_output else 'None'
    }
}

with open('repo_audit_report.json', 'w', encoding='utf-8') as rf:
    json.dump(report, rf, indent=2)

print('\n' + '='*50)
print('>>> REPO AUDIT COMPLETED <<<')
print(f'Commits: {commit_count} | Python Lines: {total_py_lines} across {len(py_files)} files')
print(f'Packaging: README={has_readme}, LICENSE={has_license}, Setup/Pyproject={has_pyproject}, CI/CD={has_workflows}')
print(f'Test Files: {len(test_files)} detected')
print(f'Potential Junk/Duplicated files: {len(untracked_junk)}')
print('Full details exported to: repo_audit_report.json')
print('='*50)
