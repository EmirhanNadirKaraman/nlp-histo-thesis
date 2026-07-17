#!/usr/bin/env python3
"""
Python Import Dependency Checker

Analyzes Python files to show import dependencies in a hierarchical structure.
Shows which files import from which other files, helping understand code relationships.

Usage:
    python scripts/relevant_file_checker.py
    python scripts/relevant_file_checker.py --file legacy/latest_ingest.py
    python scripts/relevant_file_checker.py --reverse
    python scripts/relevant_file_checker.py --external
"""

import ast
from pathlib import Path
from collections import defaultdict
from typing import Set, List, Tuple


class ImportAnalyzer:
    """Analyzes Python imports across a project."""

    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.imports = defaultdict(lambda: {'local': set(), 'external': set()})
        self.reverse_imports = defaultdict(set)
        self.python_files = []

    def find_python_files(self, exclude_dirs: Set[str] = None) -> List[Path]:
        """
        Find all Python files in the project.

        Args:
            exclude_dirs: Directories to exclude (e.g., venv, __pycache__)

        Returns:
            List of Python file paths
        """
        if exclude_dirs is None:
            exclude_dirs = {
                '.git', '__pycache__', '.pytest_cache', 'venv', 'env',
                '.venv', 'node_modules', 'build', 'dist', '.eggs',
                '*.egg-info', '.tox', '.mypy_cache'
            }

        python_files = []

        for py_file in self.project_root.rglob("*.py"):
            # Skip excluded directories
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            python_files.append(py_file)

        return sorted(python_files)

    def parse_imports(self, file_path: Path) -> Tuple[Set[str], Set[str]]:
        """
        Parse import statements from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (local_imports, external_imports)
        """
        local_imports = set()
        external_imports = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # Handle: import module
                    for alias in node.names:
                        module = alias.name
                        if self.is_local_module(module):
                            local_imports.add(module)
                        else:
                            external_imports.add(module)

                elif isinstance(node, ast.ImportFrom):
                    # Handle: from module import something
                    if node.module:
                        module = node.module
                        if self.is_local_module(module):
                            local_imports.add(module)
                        else:
                            external_imports.add(module)

        except SyntaxError as e:
            print(f"Warning: Syntax error in {file_path.relative_to(self.project_root)}: {e}")
        except Exception as e:
            print(f"Warning: Error parsing {file_path.relative_to(self.project_root)}: {e}")

        return local_imports, external_imports

    def is_local_module(self, module_name: str) -> bool:
        """
        Check if a module is part of the local project.

        Args:
            module_name: Name of the module

        Returns:
            True if module is local to the project
        """
        # Check if module corresponds to a local directory or file
        parts = module_name.split('.')
        base_module = parts[0]

        # Common local modules in this project
        local_modules = {
            'database', 'parsers', 'scripts', 'file-selector',
            'named-entity-recognition', 'misc', 'ner'
        }

        if base_module in local_modules:
            return True

        # Check if module path exists as directory or file
        module_path = self.project_root / base_module
        if module_path.exists() and module_path.is_dir():
            return True

        module_file = self.project_root / f"{base_module}.py"
        if module_file.exists():
            return True

        return False

    def analyze_all_files(self):
        """Analyze imports in all Python files."""
        print("Finding Python files...")
        self.python_files = self.find_python_files()
        print(f"Found {len(self.python_files)} Python files\n")

        print("Analyzing local imports...")
        for py_file in self.python_files:
            relative_path = str(py_file.relative_to(self.project_root))

            local_imports, external_imports = self.parse_imports(py_file)

            self.imports[relative_path]['local'] = local_imports
            self.imports[relative_path]['external'] = external_imports

            # Build reverse mapping (only for local imports)
            for local_import in local_imports:
                self.reverse_imports[local_import].add(relative_path)

        print(f"Analyzed {len(self.imports)} files\n")

    def get_module_files(self, module_name: str) -> Set[str]:
        """
        Get file paths that belong to a module.

        Args:
            module_name: Name of the module (e.g., 'database', 'parsers.xml_parsers')

        Returns:
            Set of file paths that are part of this module
        """
        module_files = set()
        module_path = module_name.replace('.', '/')

        for file_path in self.imports.keys():
            if file_path.startswith(module_path):
                module_files.add(file_path)

        return module_files

    def print_hierarchy(self, file_path: str = None, indent: int = 0, visited: Set[str] = None):
        """
        Print import hierarchy starting from a file.

        Args:
            file_path: Starting file (None for all files)
            indent: Current indentation level
            visited: Set of already visited files (to prevent cycles)
        """
        if visited is None:
            visited = set()

        if file_path is None:
            # Print all root files (files not imported by others)
            root_files = []
            imported_files = set()

            for imports_data in self.imports.values():
                for local_import in imports_data['local']:
                    module_files = self.get_module_files(local_import)
                    imported_files.update(module_files)

            for file_key in sorted(self.imports.keys()):
                if file_key not in imported_files:
                    root_files.append(file_key)

            print("="*80)
            print("Import Hierarchy (Root files → Dependencies)")
            print("="*80)

            for root_file in root_files:
                self.print_hierarchy(root_file, indent=0, visited=set())
                print()

        else:
            # Prevent infinite recursion
            if file_path in visited:
                print("  " * indent + f"└─ {file_path} (circular)")
                return

            visited.add(file_path)

            # Print current file
            prefix = "└─ " if indent > 0 else ""
            print("  " * indent + f"{prefix}{file_path}")

            # Print its imports
            if file_path in self.imports:
                local_imports = sorted(self.imports[file_path]['local'])

                for local_import in local_imports:
                    # Get all files in this module
                    module_files = self.get_module_files(local_import)

                    if module_files:
                        for module_file in sorted(module_files):
                            if module_file != file_path:  # Don't show self-imports
                                self.print_hierarchy(module_file, indent + 1, visited.copy())
                    else:
                        # Module exists but no matching files found
                        print("  " * (indent + 1) + f"└─ {local_import} (external/built-in)")

    def print_reverse_hierarchy(self, module_or_file: str = None):
        """
        Print reverse dependencies (what imports this file/module).

        Args:
            module_or_file: Module or file to check (None for all)
        """
        print("="*80)
        print("Reverse Dependencies (What imports each module)")
        print("="*80 + "\n")

        if module_or_file:
            # Check specific module/file
            if module_or_file in self.imports:
                # It's a file
                print(f"Files that import: {module_or_file}")
            else:
                # It's a module
                print(f"Files that import module: {module_or_file}")

            if module_or_file in self.reverse_imports:
                for importer in sorted(self.reverse_imports[module_or_file]):
                    print(f"  └─ {importer}")
            else:
                print("  (No files import this)")

        else:
            # Show all modules with their importers
            for module in sorted(self.reverse_imports.keys()):
                importers = self.reverse_imports[module]
                print(f"\n{module} ({len(importers)} importers):")
                for importer in sorted(importers):
                    print(f"  └─ {importer}")

    def print_external_dependencies(self):
        """Print all external dependencies used in the project."""
        all_external = set()

        for imports_data in self.imports.values():
            all_external.update(imports_data['external'])

        print("="*80)
        print(f"External Dependencies ({len(all_external)} unique)")
        print("="*80)

        # Categorize by common package groups
        categorized = defaultdict(list)

        for ext in sorted(all_external):
            base = ext.split('.')[0]
            categorized[base].append(ext)

        for base, modules in sorted(categorized.items()):
            if len(modules) == 1:
                print(f"  • {modules[0]}")
            else:
                print(f"  • {base}")
                for module in modules:
                    if module != base:
                        print(f"      └─ {module}")

    def print_file_details(self, file_path: str, show_external: bool = False):
        """
        Print detailed import information for a specific file.

        Args:
            file_path: Path to file (relative to project root)
            show_external: Whether to show external imports
        """
        if file_path not in self.imports:
            print(f"File not found: {file_path}")
            return

        print("="*80)
        print(f"Local Import Details: {file_path}")
        print("="*80 + "\n")

        local_imports = sorted(self.imports[file_path]['local'])

        print(f"Local Imports ({len(local_imports)}):")
        if local_imports:
            for imp in local_imports:
                # Show which files are part of this module
                module_files = self.get_module_files(imp)
                if module_files:
                    print(f"  • {imp}")
                    for mf in sorted(module_files)[:3]:  # Show first 3 files
                        print(f"      └─ {mf}")
                    if len(module_files) > 3:
                        print(f"      └─ ... and {len(module_files) - 3} more files")
                else:
                    print(f"  • {imp}")
        else:
            print("  (None)")

        if show_external:
            external_imports = sorted(self.imports[file_path]['external'])
            print(f"\nExternal Imports ({len(external_imports)}):")
            if external_imports:
                for imp in external_imports:
                    print(f"  • {imp}")
            else:
                print("  (None)")

        print(f"\nImported By ({len(self.reverse_imports.get(file_path, []))}):")
        if file_path in self.reverse_imports:
            for importer in sorted(self.reverse_imports[file_path]):
                print(f"  • {importer}")
        else:
            # Check if any module contains this file
            for module in self.reverse_imports:
                if file_path.startswith(module.replace('.', '/')):
                    print(f"\nModule '{module}' is imported by:")
                    for importer in sorted(self.reverse_imports[module]):
                        print(f"  • {importer}")
                    break
            else:
                print("  (Not imported by any file)")

    def print_summary(self):
        """Print summary statistics for local imports."""
        total_files = len(self.imports)
        total_local = sum(len(data['local']) for data in self.imports.values())

        # Count unique local modules
        all_local_modules = set()
        for data in self.imports.values():
            all_local_modules.update(data['local'])

        # Count files with no local imports
        files_no_imports = sum(1 for data in self.imports.values() if not data['local'])

        print("="*80)
        print("Local Import Summary")
        print("="*80)
        print(f"Total Python files:           {total_files}")
        print(f"Files with local imports:     {total_files - files_no_imports}")
        print(f"Files with no local imports:  {files_no_imports}")
        print(f"Total local import statements: {total_local}")
        print(f"Unique local modules used:    {len(all_local_modules)}")
        print("="*80 + "\n")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Python local import dependencies in the project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary of local imports
  python scripts/relevant_file_checker.py --summary

  # Show full local import hierarchy
  python scripts/relevant_file_checker.py

  # Show local imports for a specific file
  python scripts/relevant_file_checker.py --file legacy/latest_ingest.py

  # Show local imports with external dependencies
  python scripts/relevant_file_checker.py --file legacy/latest_ingest.py --show-external

  # Show reverse dependencies (what imports each local module)
  python scripts/relevant_file_checker.py --reverse

  # Show reverse dependencies for a specific module
  python scripts/relevant_file_checker.py --reverse --module database

  # Show all external dependencies
  python scripts/relevant_file_checker.py --external
        """
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Show detailed local imports for a specific file'
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Show reverse dependencies (what imports each local module)'
    )
    parser.add_argument(
        '--module',
        type=str,
        help='Filter by specific module (use with --reverse)'
    )
    parser.add_argument(
        '--external',
        action='store_true',
        help='Show all external dependencies (non-local packages)'
    )
    parser.add_argument(
        '--show-external',
        action='store_true',
        help='Include external imports in file details (use with --file)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics only'
    )

    args = parser.parse_args()

    # Get project root (parent of scripts/)
    project_root = Path(__file__).parent.parent

    # Initialize analyzer
    analyzer = ImportAnalyzer(project_root)
    analyzer.analyze_all_files()

    # Execute requested action
    if args.summary:
        analyzer.print_summary()
    elif args.file:
        analyzer.print_file_details(args.file, show_external=args.show_external)
    elif args.reverse:
        analyzer.print_reverse_hierarchy(args.module)
    elif args.external:
        analyzer.print_external_dependencies()
    else:
        # Default: show hierarchy
        analyzer.print_summary()
        analyzer.print_hierarchy()


if __name__ == "__main__":
    main()
