"""Static audit for the bug classes this project keeps hitting.

Run it directly (``python scripts/audit_static.py``); exits non-zero when a
blocking check fails, so it also works as a pre-commit or CI step.

Every defect found in the last few sessions shared a shape: something stopped
matching, and nothing said so. A CSS selector that matched no element, a
callback Input whose component had been renamed, a vendor error mapped to an
empty list, an import left behind by a refactor. None raised at the point of
breakage; all surfaced later as "the UI looks wrong".

These checks are all AST/text level, so they run in about a second and need
neither a database nor a browser.
"""

import ast
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKGS = ("layouts", "services", "models", "utils", "db", "callbacks")

findings = defaultdict(list)


def py_files():
    for p in ROOT.rglob("*.py"):
        parts = p.relative_to(ROOT).parts
        if parts[0] in {".git", "tests", "scripts", "build"}:
            continue
        if "kronos" in parts:          # vendored
            continue
        yield p


def parse(p):
    try:
        return ast.parse(p.read_text(), filename=str(p))
    except SyntaxError as e:
        findings["syntax"].append(f"{p.relative_to(ROOT)}:{e.lineno} {e.msg}")
        return None


TREES = {p: t for p in py_files() if (t := parse(p)) is not None}


# ---------------------------------------------------------------- 1. imports
def module_path(dotted):
    p = ROOT / (dotted.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = ROOT / dotted.replace(".", "/") / "__init__.py"
    return p if p.exists() else None


def exported_names(tree):
    """Top-level names a module provides to `from x import y`."""
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For)):
            for sub in ast.walk(n):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    out.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            out.add(t.id)
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    out.add(sub.target.id)
    return out


EXPORTS = {}
for path, tree in TREES.items():
    rel = path.relative_to(ROOT).with_suffix("")
    EXPORTS[".".join(rel.parts)] = exported_names(tree)

for path, tree in TREES.items():
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        mod = node.module or ""
        if not mod.startswith(PKGS) and mod not in EXPORTS:
            continue
        target = EXPORTS.get(mod)
        if target is None:
            if module_path(mod) is None:
                findings["import-module"].append(
                    f"{path.relative_to(ROOT)}:{node.lineno} no module {mod!r}")
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name not in target:
                findings["import-name"].append(
                    f"{path.relative_to(ROOT)}:{node.lineno} "
                    f"{mod}.{alias.name} does not exist")


# -------------------------------------------------------------- 2. callbacks
def dep_ids(call):
    """(kind, component_id) for each Output/Input/State in a decorator."""
    for arg in call.args:
        if not isinstance(arg, ast.Call) or not isinstance(arg.func, ast.Name):
            continue
        kind = arg.func.id
        if kind not in {"Output", "Input", "State"}:
            continue
        if not arg.args:
            continue
        target = arg.args[0]
        prop = (arg.args[1].value
                if len(arg.args) > 1 and isinstance(arg.args[1], ast.Constant)
                else "?")
        dup = any(k.arg == "allow_duplicate" for k in arg.keywords)
        opt = any(k.arg == "allow_optional" for k in arg.keywords)
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            yield kind, target.value, prop, dup, opt
        elif isinstance(target, ast.Dict):
            yield kind, None, prop, dup, opt   # pattern-matching: tolerant


CALLBACKS = []
for path, tree in TREES.items():
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = (dec.func.id if isinstance(dec.func, ast.Name)
                    else getattr(dec.func, "attr", ""))
            if name != "callback":
                continue
            deps = list(dep_ids(dec))
            kwnames = {k.arg for k in dec.keywords}
            CALLBACKS.append({
                "path": path, "line": node.lineno, "func": node.name,
                "deps": deps, "kw": kwnames, "node": node,
            })

# 2a. arity: Input+State must equal positional params
for cb in CALLBACKS:
    n_in = sum(1 for d in cb["deps"] if d[0] in ("Input", "State"))
    a = cb["node"].args
    n_args = len(a.args) + len(a.posonlyargs)
    if a.vararg or n_args == n_in:
        continue
    findings["callback-arity"].append(
        f"{cb['path'].relative_to(ROOT)}:{cb['line']} {cb['func']}() "
        f"takes {n_args} but decorator supplies {n_in}")

# 2b. duplicate Outputs (Dash errors at import unless allow_duplicate)
plain_writers = defaultdict(set)
for cb in CALLBACKS:
    here = f"{cb['path'].relative_to(ROOT)}:{cb['line']} {cb['func']}"
    for kind, cid, prop, dup, _opt in cb["deps"]:
        if kind != "Output" or cid is None or dup:
            continue
        plain_writers[(cid, prop)].add(here)
for (cid, prop), writers in plain_writers.items():
    # One writer without allow_duplicate is the primary and is correct.
    if len(writers) > 1:
        findings["duplicate-output"].append(
            f"{cid}.{prop} has {len(writers)} writers without "
            f"allow_duplicate: {', '.join(sorted(writers))}")

# 2c. ids referenced by a callback but never assigned in any layout
DEFINED = set()
ID_RE = re.compile(r"""\bid\s*=\s*["']([A-Za-z0-9_\-]+)["']""")
# id=f"{prefix}-lookback" defines a family; record the literal tail so a
# callback naming one concrete member resolves.
FSTR_ID_RE = re.compile(r"""\bid\s*=\s*f["'][^"']*\}([A-Za-z0-9_\-]+)["']""")
ID_TAILS = set()
for path in TREES:
    text = path.read_text()
    DEFINED |= set(ID_RE.findall(text))
    ID_TAILS |= set(FSTR_ID_RE.findall(text))

for cb in CALLBACKS:
    for kind, cid, prop, _dup, opt in cb["deps"]:
        if cid is None or cid in DEFINED or opt:
            continue
        if any(cid.endswith(tail) for tail in ID_TAILS):
            continue        # member of an f-string-composed id family
        findings["unknown-id"].append(
            f"{cb['path'].relative_to(ROOT)}:{cb['line']} {cb['func']} "
            f"{kind}({cid!r}) - id never assigned in any layout")


# ------------------------------------------------- 3. silent-failure funnels
SWALLOW = re.compile(
    r"except[^\n:]*:\s*\n\s*(return\s*(\[\]|\{\}|None|0|\"\"|'')\s*$|pass\s*$)",
    re.M)

# Modules whose return value becomes evidence a model or an LLM reasons over.
# Elsewhere an empty result is usually a harmless best-effort fallback; here it
# is indistinguishable from a real observation, which is how a throttled news
# fetch came to be reported as "no company-specific news" (TYL, 2026-08-04).
EVIDENCE_PATHS = (
    "services/news_service.py", "services/news_window.py",
    "services/stock_data.py", "services/bad_apples_service.py",
    "services/options_service.py", "services/analysis_runner.py",
    "models/",
)

for path in TREES:
    src = path.read_text()
    rel = str(path.relative_to(ROOT))
    for m in SWALLOW.finditer(src):
        line = src[:m.start()].count("\n") + 1
        head = m.group(0).split(chr(10))[0].strip()
        bare = re.match(r"except\s*(Exception|BaseException)?\s*(as \w+)?\s*:",
                        head)
        on_evidence = rel.startswith(EVIDENCE_PATHS)
        bucket = ("evidence-path-swallow" if (bare and on_evidence)
                  else "swallowed-exception")
        findings[bucket].append(f"{rel}:{line} {head} ...")


# --------------------------------------------------------------- 4. dead CSS
CSS_RAW = (ROOT / "assets" / "styles.css").read_text()
CSS = re.sub(r"/\*.*?\*/", "", CSS_RAW, flags=re.S)
PYTEXT = "\n".join(p.read_text() for p in TREES)
VENDOR = ("dash-", "form-", "btn", "modal", "bi-", "offcanvas", "nav",
          "col-", "row", "input-group", "card", "table", "badge", "alert",
          "toast", "tooltip", "popover", "accordion", "spinner", "js-plotly",
          "modebar", "text-", "d-", "me-", "ms-", "mb-", "mt-", "w-", "g-",
          "align-", "justify-", "flex-", "position-", "fw-", "small", "active",
          "show", "collapse", "fade", "disabled", "selected", "has-", "is-")
custom = set()
for sel in re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]{2,})", CSS):
    if not sel.startswith(VENDOR):
        custom.add(sel)
def referenced(name):
    if name in PYTEXT:
        return True
    # Built dynamically, e.g. f"home-chip-{tone}" for home-chip-positive.
    return any(name.startswith(stem) for stem in DYN_STEMS if stem)

DYN_STEMS = {m.group(1) for m in re.finditer(r'([a-z][a-z0-9-]*-)\{', PYTEXT)}
dead = sorted(s for s in custom if not referenced(s))
for s in dead[:40]:
    findings["css-unused"].append(f".{s} - not referenced from any Python file")


# ----------------------------------------------------------------- report
ORDER = ["syntax", "import-module", "import-name", "callback-arity",
         "duplicate-output", "unknown-id", "evidence-path-swallow",
         "css-unused", "swallowed-exception"]
BLOCKING = {"syntax", "import-module", "import-name", "callback-arity",
            "duplicate-output"}

print(f"scanned {len(TREES)} modules, {len(CALLBACKS)} callbacks\n")
total_blocking = 0
for key in ORDER:
    items = findings.get(key, [])
    if not items:
        print(f"  [ok]   {key}")
        continue
    tag = "FAIL" if key in BLOCKING else "note"
    if key in BLOCKING:
        total_blocking += len(items)
    print(f"  [{tag}] {key}: {len(items)}")
    for it in items[:12]:
        print(f"           {it}")
    if len(items) > 12:
        print(f"           ... and {len(items) - 12} more")

print(f"\nblocking findings: {total_blocking}")
sys.exit(1 if total_blocking else 0)
