"""Project-wide C++ identifier renamer built on libclang.

Naming rules:
  functions / methods                  -> snake_case
  classes / structs / enums / aliases  -> PascalCase
  variables / parameters / fields      -> camelCase

Names that already follow their rule are left alone, so acronyms such as
NNOutput are kept. ALL_CAPS names (constants, enum values) are never touched.

It works in two passes over every translation unit:
  1. Collect each declaration *and every reference to it* (calls, member
     accesses, type uses, constructor init lists, ...) inside the project
     files, keyed by the declaration's USR (libclang's unique symbol id).
  2. Rewrite every collected occurrence in every file (headers included),
     using byte offsets.

Usage (from the repo root or anywhere):
  python convert.py                 # dry run: print the planned renames
  python convert.py --apply         # rewrite the files in place
  python convert.py --src ../src --libclang /usr/lib/llvm-18/lib/libclang.so

The python bindings must match the libclang version, e.g. for LLVM 18:
  pip install "clang==18.*"
Commit your work before running with --apply.
"""

# pyright: reportAttributeAccessIssue=false
import argparse
import ctypes
import json
import os
import re
import sys
from collections import defaultdict

import clang.cindex as ci

K = ci.CursorKind

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DISPLAY_ROOT = REPO_ROOT  # paths in the report are shown relative to this

DEFAULT_LIBCLANG = "/usr/lib/llvm-18/lib/libclang.so"
DEFAULT_FLAGS = [
    "-x", "c++", "-std=c++20",
    "-I/home/me/libtorch/include",
    "-I/home/me/libtorch/include/torch/csrc/api/include",
]
SOURCE_EXTS = (".cpp", ".cc", ".cxx")
HEADER_EXTS = (".h", ".hpp", ".hh")
EXCLUDE_DIRS = {"nlohmann"}          # third-party code living under src/
KEEP_NAMES = {"main", "json"}        # never rename these

FUNCTION_KINDS = {K.FUNCTION_DECL, K.CXX_METHOD, K.FUNCTION_TEMPLATE}
TYPE_KINDS = {K.CLASS_DECL, K.STRUCT_DECL, K.UNION_DECL, K.ENUM_DECL,
              K.CLASS_TEMPLATE, K.TYPEDEF_DECL, K.TYPE_ALIAS_DECL,
              K.TYPE_ALIAS_TEMPLATE_DECL}
VARIABLE_KINDS = {K.VAR_DECL, K.PARM_DECL, K.FIELD_DECL}
# Cursors that point at a declaration through .referenced
REFERENCE_KINDS = {K.DECL_REF_EXPR, K.MEMBER_REF_EXPR, K.TYPE_REF,
                   K.TEMPLATE_REF, K.MEMBER_REF, K.VARIABLE_REF,
                   K.OVERLOADED_DECL_REF}

IDENT = re.compile(rb"[A-Za-z0-9_]")


# --------------------------------------------------------------------------
# Case conversion
# --------------------------------------------------------------------------

def _split_affixes(name):
  """'_foo_' -> ('_', 'foo', '_'): leading/trailing underscores are kept."""
  core = name.strip("_")
  lead = name[:len(name) - len(name.lstrip("_"))]
  trail = name[len(name.rstrip("_")):]
  return lead, core, trail


def is_constant_style(name):
  core = name.strip("_")
  return bool(core) and core.upper() == core and any(c.isalpha() for c in core)


def to_snake_case(name):
  lead, core, trail = _split_affixes(name)
  s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", core)
  s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
  s = re.sub(r"_+", "_", s).lower()
  return lead + s + trail


def _lower_head(word):
  """Lowercase the leading capital run: Value->value, NNValue->nnValue, NN->nn."""
  m = re.match(r"[A-Z]+", word)
  if not m:
    return word
  k = len(m.group())
  if k > 1 and k < len(word) and word[k].islower():
    k -= 1  # keep the capital that starts the next word
  return word[:k].lower() + word[k:]


def _upper_head(word):
  return word[:1].upper() + word[1:]


def to_camel_case(name):
  lead, core, trail = _split_affixes(name)
  parts = [p for p in core.split("_") if p]
  if not parts:
    return name
  return lead + _lower_head(parts[0]) + "".join(_upper_head(p) for p in parts[1:]) + trail


def to_pascal_case(name):
  lead, core, trail = _split_affixes(name)
  parts = [p for p in core.split("_") if p]
  if not parts:
    return name
  return lead + "".join(_upper_head(p) for p in parts) + trail


SNAKE_RE = re.compile(r"^_*[a-z][a-z0-9_]*$")
CAMEL_RE = re.compile(r"^_*[a-z][A-Za-z0-9]*_*$")
PASCAL_RE = re.compile(r"^_*[A-Z][A-Za-z0-9]*_*$")


def wanted_name(cursor):
  """New name for a declaration cursor, or None to leave it alone."""
  name = cursor.spelling
  if not name or name in KEEP_NAMES or is_constant_style(name):
    return None
  kind = cursor.kind
  if kind in FUNCTION_KINDS:
    if name.startswith("operator") or overrides_external(cursor):
      return None
    return None if SNAKE_RE.match(name) else to_snake_case(name)
  if kind in TYPE_KINDS:
    return None if PASCAL_RE.match(name) else to_pascal_case(name)
  if kind in VARIABLE_KINDS:
    return None if CAMEL_RE.match(name) else to_camel_case(name)
  return None


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------

PROJECT_FILES = set()  # realpaths of the files we are allowed to touch


_REALPATH_CACHE = {}


def file_of(cursor):
  f = cursor.location.file
  if not f:
    return None
  name = f.name
  path = _REALPATH_CACHE.get(name)
  if path is None:
    path = _REALPATH_CACHE[name] = os.path.realpath(name)
  return path


def in_project(cursor):
  return file_of(cursor) in PROJECT_FILES


def text_location(node):
  """(realpath, byte offset) where the node's name is actually written.

  node.location gives the macro *expansion* site, so for `assert(isLegal(m))`
  it points at `assert`. clang_getFileLocation follows macro arguments back
  to the token in the source (and into the macro definition for tokens the
  macro itself generates, which then fall outside the project and are skipped)."""
  fn = ci.conf.lib.clang_getFileLocation
  fn.argtypes = [ci.SourceLocation, ctypes.POINTER(ci.c_object_p),
                 ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                 ctypes.POINTER(ctypes.c_uint)]
  fn.restype = None
  f, line, col, off = ci.c_object_p(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
  fn(node.location, ctypes.byref(f), ctypes.byref(line), ctypes.byref(col), ctypes.byref(off))
  if not f:
    return None, 0
  name = ci.File(f).name
  path = _REALPATH_CACHE.get(name)
  if path is None:
    path = _REALPATH_CACHE[name] = os.path.realpath(name)
  return path, off.value


def overloaded_decls(ref):
  """Candidate declarations of an OVERLOADED_DECL_REF cursor."""
  lib = ci.conf.lib
  num = lib.clang_getNumOverloadedDecls
  num.argtypes, num.restype = [ci.Cursor], ctypes.c_uint
  get = lib.clang_getOverloadedDecl
  get.argtypes, get.restype = [ci.Cursor, ctypes.c_uint], ci.Cursor
  result = []
  for i in range(num(ref)):
    c = get(ref, i)
    c._tu = ref._tu
    if not c.kind.is_invalid():
      result.append(c)
  return result


def overridden_cursors(method):
  """Methods directly overridden by `method` (older bindings lack a wrapper)."""
  if hasattr(method, "get_overridden_cursors"):
    return list(method.get_overridden_cursors() or [])
  lib = ci.conf.lib
  fn = lib.clang_getOverriddenCursors
  fn.argtypes = [ci.Cursor, ctypes.POINTER(ctypes.POINTER(ci.Cursor)),
                 ctypes.POINTER(ctypes.c_uint)]
  fn.restype = None
  arr, n = ctypes.POINTER(ci.Cursor)(), ctypes.c_uint()
  fn(method, ctypes.byref(arr), ctypes.byref(n))
  result = []
  for i in range(n.value):
    c = ci.Cursor()
    ctypes.memmove(ctypes.byref(c), ctypes.byref(arr[i]), ctypes.sizeof(ci.Cursor))
    c._tu = method._tu
    result.append(c)
  if n.value:
    lib.clang_disposeOverriddenCursors(arr)
  return result


def overrides_external(method):
  """True if a virtual method overrides one declared outside the project
  (e.g. torch::nn::Module), in which case its name is not ours to change."""
  if method.kind != K.CXX_METHOD:
    return False
  stack = overridden_cursors(method)
  while stack:
    base = stack.pop()
    if not in_project(base):
      return True
    stack.extend(overridden_cursors(base))
  return False


def key_of(decl):
  decl = decl.canonical
  usr = decl.get_usr()
  if usr:
    return usr
  return f"{file_of(decl)}:{decl.location.offset}:{decl.spelling}"


def target_decl(node):
  """(declaration cursor naming this token, byte skip) or (None, 0).

  Constructors and destructors are spelled with the class name, so they map
  to the class itself. The skip of 1 steps over the '~' of a destructor."""
  kind = node.kind
  if kind.is_declaration():
    decl = node
  elif kind in REFERENCE_KINDS:
    decl = node.referenced
    if kind == K.OVERLOADED_DECL_REF:
      decl = node
  else:
    return None, 0
  if decl is None:
    return None, 0
  if decl.kind == K.OVERLOADED_DECL_REF:
    # Unresolved call inside a template, e.g. `reflectHorizontal(m, 1)` with a
    # dependent argument: pick a candidate declared in the project.
    candidates = overloaded_decls(decl)
    ours = [c for c in candidates if in_project(c)]
    decl = (ours or candidates or [None])[0]
    if decl is None:
      return None, 0
  if decl.kind in (K.CONSTRUCTOR, K.DESTRUCTOR):
    skip = 1 if decl.kind == K.DESTRUCTOR and node.kind.is_declaration() else 0
    parent = decl.semantic_parent
    return (parent, skip) if parent is not None else (None, 0)
  return decl, 0


# --------------------------------------------------------------------------
# Pass 1: collect
# --------------------------------------------------------------------------

class Collector:
  def __init__(self):
    self.renames = {}                    # key -> (old, new, kind, "file:line")
    self.skipped = {}                    # key -> reason
    self.occurrences = defaultdict(set)  # file -> {(offset, key)}
    self.decl_names = {}                 # key -> original spelling (collision check)

  def visit_tu(self, tu):
    # Only descend into top-level declarations that live in project files;
    # walking all of libtorch's AST would take minutes per file.
    for top in tu.cursor.get_children():
      if in_project(top):
        self.visit_tree(top)

  def visit_tree(self, root):
    for node in root.walk_preorder():
      if node.location.file is None or not in_project(node):
        continue
      decl, skip = target_decl(node)
      if decl is None or not decl.spelling:
        continue
      key = key_of(decl)
      if decl.kind.is_declaration() and in_project(decl):
        self.decl_names[key] = decl.spelling
      if key not in self.renames and key not in self.skipped:
        self.classify(decl, key)
      if key in self.renames:
        path, offset = text_location(node)
        if path in PROJECT_FILES:
          self.occurrences[path].add((offset + skip, key))

  def classify(self, decl, key):
    if not in_project(decl.canonical) and not in_project(decl):
      self.skipped[key] = "declared outside project"
      return
    new = wanted_name(decl)
    if new is None or new == decl.spelling:
      self.skipped[key] = "already conforms / excluded"
      return
    where = f"{os.path.relpath(file_of(decl) or "?", DISPLAY_ROOT)}:{decl.location.line}"
    self.renames[key] = (decl.spelling, new, decl.kind.name, where)


# --------------------------------------------------------------------------
# Pass 2: rewrite
# --------------------------------------------------------------------------

def plan_edits(collector):
  """file -> sorted [(offset, old, new)], plus a list of mismatch warnings."""
  edits, warnings = {}, []
  for path, occs in collector.occurrences.items():
    with open(path, "rb") as f:
      data = f.read()
    seen, todo = set(), []
    for off, key in occs:
      if off in seen:
        continue
      old, new, _, _ = collector.renames[key]
      ob = old.encode()
      end = off + len(ob)
      whole_word = (data[off:end] == ob
                    and (off == 0 or not IDENT.match(data[off - 1:off]))
                    and (end >= len(data) or not IDENT.match(data[end:end + 1])))
      if not whole_word:
        # Usually a macro expansion (e.g. TORCH_MODULE); the token isn't there.
        line = data[:off].count(b"\n") + 1
        warnings.append(f"{os.path.relpath(path, DISPLAY_ROOT)}:{line}: expected '{old}', "
                        f"found '{data[off:end].decode(errors='replace')}' - skipped")
        continue
      seen.add(off)
      todo.append((off, old, new))
    edits[path] = sorted(todo)
  return edits, warnings


def apply_edits(edits):
  for path, todo in edits.items():
    if not todo:
      continue
    with open(path, "rb") as f:
      data = f.read()
    for off, old, new in reversed(todo):  # back to front keeps offsets valid
      data = data[:off] + new.encode() + data[off + len(old.encode()):]
    with open(path, "wb") as f:
      f.write(data)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def find_project_files(src_dir):
  sources, headers = [], []
  for root, dirs, files in os.walk(src_dir):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for fn in sorted(files):
      p = os.path.realpath(os.path.join(root, fn))
      if fn.endswith(SOURCE_EXTS):
        sources.append(p)
      elif fn.endswith(HEADER_EXTS):
        headers.append(p)
  return sources, headers


def load_compile_commands(path):
  """file realpath -> compiler args (minus the compiler, -c/-o and the file)."""
  if not path or not os.path.exists(path):
    return {}
  out = {}
  with open(path) as f:
    for entry in json.load(f):
      args = entry.get("arguments") or entry["command"].split()
      file = os.path.realpath(os.path.join(entry["directory"], entry["file"]))
      cleaned, skip_next = [], False
      for a in args[1:]:
        if skip_next:
          skip_next = False
          continue
        if a in ("-o", "-c"):
          skip_next = a == "-o"
          continue
        if os.path.realpath(os.path.join(entry["directory"], a)) == file:
          continue
        if a.startswith("-I") and len(a) > 2 and not os.path.isabs(a[2:]):
          a = "-I" + os.path.join(entry["directory"], a[2:])
        cleaned.append(a)
      out[file] = cleaned
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                  help="source directory to convert (default: ../src)")
  ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
  ap.add_argument("--force", action="store_true",
                  help="apply even if some files failed to parse cleanly")
  ap.add_argument("--libclang", default=DEFAULT_LIBCLANG,
                  help="path to libclang.so ('' to use the default search)")
  ap.add_argument("--compile-commands",
                  default=os.path.join(REPO_ROOT, "build", "compile_commands.json"),
                  help="compile_commands.json to take flags from, if it exists")
  ap.add_argument("--flag", action="append", default=[],
                  help="extra compiler flag (repeatable)")
  args = ap.parse_args()

  if args.libclang:
    ci.Config.set_library_file(args.libclang)

  global DISPLAY_ROOT
  src_dir = os.path.realpath(args.src)
  DISPLAY_ROOT = os.path.dirname(src_dir)
  sources, headers = find_project_files(src_dir)
  if not sources and not headers:
    sys.exit(f"No C++ files found under {src_dir}")
  PROJECT_FILES.update(sources + headers)

  ccmds = load_compile_commands(args.compile_commands)
  default_flags = DEFAULT_FLAGS + [f"-I{src_dir}"] + args.flag

  index = ci.Index.create()
  collector = Collector()
  seen_files, broken = set(), []

  def parse(path, flags):
    rel = os.path.relpath(path, DISPLAY_ROOT)
    print(f"parsing {rel} ...", file=sys.stderr)
    tu = index.parse(path, args=flags)
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
      broken.append(rel)
      for d in errors[:5]:
        print(f"  error: {d}", file=sys.stderr)
    collector.visit_tu(tu)
    for inc in tu.get_includes():
      seen_files.add(os.path.realpath(inc.include.name))
    seen_files.add(path)

  for path in sources:
    flags = ccmds.get(path, default_flags)
    if path in ccmds:
      flags = ["-x", "c++"] + flags + args.flag
    parse(path, flags)
  # Headers that no .cpp includes still need converting.
  for path in headers:
    if path not in seen_files:
      parse(path, default_flags)

  edits, warnings = plan_edits(collector)

  by_kind = defaultdict(list)
  for old, new, kind, where in collector.renames.values():
    by_kind[kind].append((where, old, new))
  for kind in sorted(by_kind):
    print(f"\n== {kind} ==")
    for where, old, new in sorted(set(by_kind[kind])):
      print(f"  {old:30s} -> {new:30s} ({where})")

  # Two different symbols ending up with the same name may clash.
  # A new name is suspicious only if some symbol that is *not* being renamed
  # already has it. (Swaps such as field model_type -> modelType alongside
  # method modelType -> model_type are fine, as are overloads.)
  unchanged = {name for key, name in collector.decl_names.items()
               if key not in collector.renames}
  collisions = sorted({new for _, new, _, _ in collector.renames.values()
                       if new in unchanged})
  if collisions:
    print("\nPossible collisions (new name already used by a symbol that is not "
          "renamed; fine if they live in different scopes):")
    for n in sorted(collisions):
      print(f"  {n}")

  if warnings:
    print("\nSkipped occurrences:")
    for w in warnings:
      print(f"  {w}")

  total = sum(len(v) for v in edits.values())
  print(f"\n{len(collector.renames)} symbols, {total} occurrences in "
        f"{sum(1 for v in edits.values() if v)} files.")
  if broken:
    print(f"WARNING: {len(broken)} file(s) had parse errors, so some references may "
          f"be missing: {', '.join(broken)}")

  if not args.apply:
    print("Dry run - nothing written. Re-run with --apply to rewrite files.")
    return
  if broken and not args.force:
    sys.exit("Refusing to apply because of parse errors (use --force to override).")
  apply_edits(edits)
  print("Files rewritten. Rebuild to check: cd build && make")


if __name__ == "__main__":
  main()
