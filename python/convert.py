import re
import sys
import clang.cindex


def to_snake_case(name):
  """Converts any case to snake_case."""
  if not name:
    return name
  # Handle existing camel/Pascal casing
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
  s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
  return s2.replace("__", "_").lower()


def to_camel_case(name):
  """Converts any case to camelCase."""
  if not name:
    return name
  # First normalize to snake_case to get clean word boundaries
  snake = to_snake_case(name)
  components = [c for c in snake.split("_") if c]
  if not components:
    return name
  return components[0] + "".join(x.capitalize() for x in components[1:])


def refactor_cpp_file(file_path):
  # Initialize Clang Index
  index = clang.cindex.Index.create()

  # Parse the C++ file (adjust arguments like -std=c++17 if needed)
  tu = index.parse(file_path, args=["-std=c++17"])

  # Array to keep track of text replacements: (start_offset, end_offset, new_text)
  replacements = []

  # Traverse the AST node tree
  def traverse(node):
    # Only modify tokens belonging strictly to the source file (skip includes)
    if node.location.file and node.location.file.name == file_path:

      # Determine if node needs a name change
      new_name = None

      # 1. Convert Function Names to snake_case
      if node.kind in (
          clang.cindex.CursorKind.FUNCTION_DECL,
          clang.cindex.CursorKind.CXX_METHOD,
      ):
        # Ignore operators and constructors/destructors
        if not (
            node.spelling.startswith("operator")
            or node.spelling.startswith("~")
            or node.spelling == node.semantic_parent.spelling
        ):
          new_name = to_snake_case(node.spelling)

      # 2. Convert Variables, Parameters, Classes, and Structs to camelCase
      elif node.kind in (
          clang.cindex.CursorKind.VAR_DECL,
          clang.cindex.CursorKind.PARM_DECL,
          clang.cindex.CursorKind.CLASS_DECL,
          clang.cindex.CursorKind.STRUCT_DECL,
          clang.cindex.CursorKind.FIELD_DECL,
      ):
        new_name = to_camel_case(node.spelling)

      # If a rule was matched and the name actually changes, record it
      if new_name and new_name != node.spelling:
        # Get token extent offsets inside the source file
        start = node.location.offset
        end = start + len(node.spelling)
        replacements.append((start, end, new_name))

    # Recurse down to child tokens
    for child in node.get_children():
      traverse(child)

  traverse(tu.cursor)

  # Sort replacements in reverse order so modifying the file string
  # from back-to-front doesn't break target index offsets
  replacements.sort(key=lambda x: x[0], reverse=True)

  # Read, apply changes, and rewrite file
  with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

  distinct_replacements = []
  seen_starts = set()
  for r in replacements:
    if r[0] not in seen_starts:
      seen_starts.add(r[0])
      distinct_replacements.append(r)

  for start, end, new_name in distinct_replacements:
    # Quick sanity check to ensure we are replacing the intended name token
    if content[start:end].isalnum() or "_" in content[start:end]:
      content = content[:start] + new_name + content[end:]

  with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

  print(f"Successfully automated renaming for: {file_path}")


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python refactor.py <path_to_cpp_file.cpp>")
  else:
    refactor_cpp_file(sys.argv[1])
