"""
security_check.py - read-only security audit of the CropSense project.

Run from the project folder (with the venv active):

    python security_check.py

It never changes or deletes anything. Exit code 1 means at least one FAIL.
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"venv", ".venv", "env", ".git", "__pycache__", "node_modules", "datasets", "models", ".idea", ".vscode"}
TEXT_EXT = {".py", ".js", ".html", ".css", ".json", ".txt", ".md", ".env", ".cfg", ".ini", ".yml", ".yaml", ".toml", ".csv"}
THIS_FILE = os.path.abspath(__file__)

SECRET_PATTERNS = {
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    "hard-coded secret assignment": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret[_-]?key|secret|token|passwd|password|pwd)\b\s*[=:]\s*["'][^"'\s]{8,}["']"""),
}
SENSITIVE_NAMES = re.compile(r"(?i)(^\.env(\..*)?$|\.pem$|\.key$|\.pfx$|\.p12$|^id_rsa|^id_ed25519|credentials.*\.json$|\.sqlite3?$|\.db$)")

results = []   # (level, message)


def add(level, message):
    results.append((level, message))
    print(f"[{level}] {message}")


def walk_project():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield os.path.join(dirpath, name)


def rel(path):
    return os.path.relpath(path, ROOT)


# 1. secrets in files
def check_files():
    leaks, sensitive = [], []
    for path in walk_project():
        if os.path.abspath(path) == THIS_FILE:
            continue
        name = os.path.basename(path)
        if SENSITIVE_NAMES.search(name):
            sensitive.append(rel(path))
        if os.path.splitext(name)[1].lower() in TEXT_EXT or name.startswith(".env"):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for n, line in enumerate(f, 1):
                        for label, rx in SECRET_PATTERNS.items():
                            if rx.search(line):
                                leaks.append(f"{rel(path)}:{n} looks like a {label}")
            except OSError:
                pass
    if leaks:
        for item in leaks:
            add("FAIL", item)
    else:
        add("PASS", "No keys, tokens or hard-coded passwords found in project files.")
    if sensitive:
        for item in sensitive:
            add("WARN", f"Sensitive-looking file in the project: {item}")
    else:
        add("PASS", "No .env, key, certificate or database files in the project folder.")


# 2. debug mode and app settings
def check_app_settings():
    app_path = os.path.join(ROOT, "app.py")
    if not os.path.exists(app_path):
        add("WARN", "app.py not found next to this script.")
        return
    text = open(app_path, encoding="utf-8", errors="ignore").read()
    if re.search(r"debug\s*=\s*True", text) or re.search(r"(?i)FLASK_DEBUG\s*=\s*1", text):
        add("FAIL", "Debug mode is switched on in app.py (debug=True).")
    else:
        add("PASS", "Debug mode is off in app.py.")
    if os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "on"):
        add("FAIL", "The FLASK_DEBUG environment variable is on in this terminal.")
    for needle, label in [("Content-Security-Policy", "Content Security Policy header"),
                          ("MAX_CONTENT_LENGTH", "upload size limit"),
                          ("RateLimiter", "rate limiting")]:
        add("PASS" if needle in text else "FAIL", f"{label} {'is' if needle in text else 'is NOT'} present in app.py.")
    if re.search(r"(?i)flask_cors|CORS\(", text):
        add("WARN", "CORS is enabled in app.py. Make sure only the origins you trust are allowed.")
    else:
        add("PASS", "CORS is not enabled (other websites cannot read your API responses).")
    if bool(os.environ.get("APP_USER")) != bool(os.environ.get("APP_PASSWORD_HASH")):
        add("WARN", "Only one of APP_USER / APP_PASSWORD_HASH is set, so password protection stays OFF.")
    if os.environ.get("APP_PASSWORD_HASH") and not os.environ["APP_PASSWORD_HASH"].startswith(("scrypt:", "pbkdf2:")):
        add("FAIL", "APP_PASSWORD_HASH is not a proper hash. It must start with scrypt: or pbkdf2:.")


# 3. exposed files
def check_exposure():
    static = os.path.join(ROOT, "static")
    if os.path.isdir(static):
        odd = []
        for dirpath, _, files in os.walk(static):
            for name in files:
                if os.path.splitext(name)[1].lower() not in {".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2"}:
                    odd.append(os.path.relpath(os.path.join(dirpath, name), ROOT))
        if odd:
            for item in odd[:10]:
                add("WARN", f"Unexpected file in static/ (served publicly): {item}")
        else:
            add("PASS", "static/ only contains CSS, JS and images.")
        uploads = os.path.join(static, "uploads")
        if os.path.isdir(uploads) and os.listdir(uploads):
            add("WARN", "static/uploads holds old photos from the previous version. They are publicly reachable by name. Delete that folder.")
        else:
            add("PASS", "No uploaded photos are stored on disk.")
    gi = os.path.join(ROOT, ".gitignore")
    if not os.path.exists(gi):
        add("WARN", "No .gitignore file. Add one so venv, .env and uploads never get committed.")
    else:
        g = open(gi, encoding="utf-8", errors="ignore").read()
        missing = [x for x in ("venv", ".env", "__pycache__") if x not in g]
        add("WARN" if missing else "PASS", ".gitignore is missing: " + ", ".join(missing) if missing else ".gitignore covers venv, .env and __pycache__.")


# 4. git history
def check_git():
    git = shutil.which("git")
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        add("INFO", "This folder is not a git repository, so there is no history to leak from.")
        return
    if not git:
        add("WARN", "A .git folder exists but git is not installed or not on PATH, so history was not scanned.")
        return
    names = subprocess.run([git, "log", "--all", "--name-only", "--pretty=format:"], cwd=ROOT,
                           capture_output=True, text=True, errors="ignore").stdout.splitlines()
    bad_files = sorted({n for n in names if n and SENSITIVE_NAMES.search(os.path.basename(n))})
    for n in bad_files:
        add("FAIL", f"Sensitive-looking file was committed at some point: {n}")
    proc = subprocess.Popen([git, "log", "--all", "-p", "--no-color", "--pretty=format:commit %h"], cwd=ROOT,
                            stdout=subprocess.PIPE, text=True, errors="ignore")
    commit, found, scanned = "?", [], 0
    for line in proc.stdout:
        scanned += 1
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.startswith("+") and not line.startswith("+++"):
            for label, rx in SECRET_PATTERNS.items():
                if rx.search(line):
                    found.append(f"commit {commit}: {label}")
        if scanned > 5_000_000:
            add("WARN", "History is very large; scan stopped after 5 million lines.")
            proc.kill()
            break
    proc.wait()
    if found:
        for item in sorted(set(found))[:20]:
            add("FAIL", f"Possible secret in git history: {item}")
        add("INFO", "If a real secret was committed, REVOKE it at the provider first. Deleting it from history is not enough.")
    elif not bad_files:
        add("PASS", "No secrets or sensitive files found in the git history.")


# 5. dependencies
def check_dependencies():
    try:
        import pip_audit  # noqa: F401
    except ImportError:
        add("WARN", "pip-audit is not installed, so packages were NOT checked for known vulnerabilities. "
                    "Run:  pip install pip-audit   then   pip-audit")
        return
    proc = subprocess.run([sys.executable, "-m", "pip_audit", "--progress-spinner", "off"], cwd=ROOT,
                          capture_output=True, text=True, errors="ignore")
    if proc.returncode == 0:
        add("PASS", "pip-audit found no known vulnerabilities in installed packages.")
    else:
        add("FAIL", "pip-audit reported problems. Details below:")
        print((proc.stdout or proc.stderr)[-3000:])


def main():
    print(f"Auditing: {ROOT}\n")
    check_files()
    check_app_settings()
    check_exposure()
    check_git()
    check_dependencies()
    counts = {k: sum(1 for lvl, _ in results if lvl == k) for k in ("PASS", "INFO", "WARN", "FAIL")}
    print(f"\nSummary: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed.")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()