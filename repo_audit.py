import os
import subprocess
from datetime import datetime

def run_cmd(args):
try:
p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="ignore")
return p.stdout.strip()
except Exception:
return ""

def get_tree(path, prefix=""):
lines = []
try:
entries = sorted([e for e in os.listdir(path) if e != ".git"])
except Exception:
return lines
for i, entry in enumerate(entries):
is_last = (i == len(entries) - 1)
pointer = "└── " if is_last else "├── "
lines.append(f"{prefix}{pointer}{entry}")
full = os.path.join(path, entry)
if os.path.isdir(full):
lines.extend(get_tree(full, prefix + ("    " if is_last else "│   ")))
return lines

def main():
if not os.path.exists(".git"):
print("[-] Error: Not a git repository.")
return

repo_name = os.path.basename(os.path.abspath("."))
branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
branches = [b.strip() for b in run_cmd(["git", "branch", "-a"]).splitlines() if b.strip()]

# لاگ کامل کامیت‌ها
raw_log = run_cmd(["git", "log", "--pretty=format:%h|%ad|%an|%s", "--date=short", "--shortstat"])
commits = []
cur = None
total_added = 0
total_deleted = 0

for line in raw_log.splitlines():
line = line.strip()
if not line:
continue
if "|" in line and len(line.split("|")) >= 4:
parts = line.split("|", 3)
cur = {
"hash": parts[0],
"date": parts[1],
"author": parts[2],
"subject": parts[3],
"stat": "No file stats"
}
commits.append(cur)
elif cur and ("insertion" in line or "deletion" in line or "file changed" in line):
cur["stat"] = line
# شمارش تخمینی خطوط
for part in line.split(","):
part = part.strip()
if "insertion" in part:
total_added += int(part.split()[0])
elif "deletion" in part:
total_deleted += int(part.split()[0])

commit_count = len(commits)
first_commit = commits[-1]["date"] if commits else "N/A"
last_commit = commits[0]["date"] if commits else "N/A"

# بررسی ساختار پوشه‌ها
has_readme = any("readme" in f.lower() for f in os.listdir("."))
has_docs = os.path.exists("docs") or any("doc" in f.lower() for f in os.listdir("."))
has_tests = os.path.exists("tests") or os.path.exists("test")
has_ci = os.path.exists(".github")

# نمره‌دهی شاخص‌ها (مجموع ۱۰۰)
score_commit = min(30, round(commit_count * 0.5, 1))   # تا ۳۰ نمره
score_struct = (15 if has_readme else 0) + (10 if has_docs else 0) + (10 if has_tests else 0) + (5 if has_ci else 0) # تا ۴۰ نمره
score_vol = min(20, (total_added // 100) * 2)          # تا ۲۰ نمره
score_branch = min(10, len(branches) * 2)               # تا ۱۰ نمره
total_score = min(100, score_commit + score_struct + score_vol + score_branch)

status = "عالی و استاندارد (Professional)" if total_score >= 75 else "رو به رشد و خوب (Growing Steady)" if total_score >= 45 else "مقدماتی (Needs Setup)"

# نگارش گزارش Markdown
md = []
md.append(f"# گزارش جامع ریپازیتوری: {repo_name}")
md.append(f"- زمان گزارش: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
md.append("")
md.append("## ۱. امتیاز ارزیابی و سطح پیشرفت")
md.append(f"- **امتیاز کل:** `{total_score}/100` ({status})")
md.append(f"  - فعالیت کامیت‌ها: `{score_commit}/30` (تعداد کل کامیت‌ها: {commit_count})")
md.append(f"  - مستندات و تست و CI: `{score_struct}/40` (README: {has_readme}, Docs: {has_docs}, Tests: {has_tests}, CI: {has_ci})")
md.append(f"  - حجم کد و تغییرات: `{score_vol}/20` (اضافه شده: +{total_added} خط، حذف شده: -{total_deleted} خط)")
md.append(f"  - ساختار شاخه‌ها: `{score_branch}/10` (تعداد شاخه‌ها: {len(branches)})")
md.append("")
md.append("## ۲. متادیتای مخزن")
md.append(f"- شاخه جاری: `{branch}`")
md.append(f"- تاریخ اولین کامیت: `{first_commit}`")
md.append(f"- تاریخ آخرین کامیت: `{last_commit}`")
md.append("")
md.append("## ۳. ساختار درختی ۰ تا ۱۰۰ ریپازیتوری")
md.append("
```text")
    md.append(".")
    md.extend(get_tree("."))
    md.append("
```")
md.append("")
md.append("## ۴. تاریخچه کامل کامیت‌ها")
for c in commits:
md.append(f"### [{c['hash']}] {c['date']} - {c['author']}")
md.append(f"> {c['subject']}")
md.append(f"- آمار: `{c['stat']}`")
md.append("")

with open("repo_report.md", "w", encoding="utf-8") as f:
f.write("\n".join(md))

print("\n" + "="*50)
print(f" امتیاز کل فعالیت: {total_score}/100 -> {status}")
print(f" تعداد کامیت‌ها: {commit_count} | خطوط اضافه شده: +{total_added}")
print(" گزارش کامل در فایل repo_report.md ذخیره شد.")
print("="*50 + "\n")

if __name__ == "__main__":
main()
