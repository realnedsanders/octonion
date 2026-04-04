"""Auto-generate API reference pages from source modules.

This script is executed at build time by mkdocs-gen-files. It walks
src/octonion/, creates a markdown page per module with a single
mkdocstrings ::: directive, and builds the nav structure for
mkdocs-literate-nav.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
src = Path("src")

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__, __init__, and private subpackage internals
    if "__pycache__" in parts:
        continue
    # Show __init__ as the package-level page
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    # Skip private modules in subpackages (baselines/_trainer.py etc.)
    # but keep top-level private modules (_octonion.py, _fano.py etc.)
    elif parts[-1].startswith("_") and len(parts) > 2:
        continue

    identifier = ".".join(parts)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
