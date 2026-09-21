"""
Setup script to initialize the scientific modular directory structure
for CCUS Adsorption Simulation Framework.
"""

from pathlib import Path

# Base project root
ROOT_DIR = Path(".")

# Define folders to create
DIRECTORIES = [
    ROOT_DIR / "configs",
    ROOT_DIR / "data" / "experimental",
    ROOT_DIR / "data" / "literature_benchmarks",
    ROOT_DIR / "notebooks",
    ROOT_DIR / "tests",
    ROOT_DIR / "src" / "thermodynamics",
    ROOT_DIR / "src" / "kinetics",
    ROOT_DIR / "src" / "hydrodynamics",
    ROOT_DIR / "src" / "models",
    ROOT_DIR / "src" / "solvers",
    ROOT_DIR / "src" / "cycles",
    ROOT_DIR / "src" / "utils",
]

# Define initial blank/template files
INIT_FILES = [
    ROOT_DIR / "src" / "__init__.py",
    ROOT_DIR / "src" / "thermodynamics" / "__init__.py",
    ROOT_DIR / "src" / "kinetics" / "__init__.py",
    ROOT_DIR / "src" / "hydrodynamics" / "__init__.py",
    ROOT_DIR / "src" / "models" / "__init__.py",
    ROOT_DIR / "src" / "solvers" / "__init__.py",
    ROOT_DIR / "src" / "cycles" / "__init__.py",
    ROOT_DIR / "src" / "utils" / "__init__.py",
    ROOT_DIR / "tests" / "__init__.py",
    ROOT_DIR / "main.py",
    ROOT_DIR / "requirements.txt",
    ROOT_DIR / "README.md",
]


def setup():
    # 1. Create Directories
    for folder in DIRECTORIES:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[CREATED DIR]  {folder}")

    # 2. Create __init__ and core files
    for file_path in INIT_FILES:
        if not file_path.exists():
            file_path.touch()
            print(f"[CREATED FILE] {file_path}")

    print("\nProject structure created successfully!")


if __name__ == "__main__":
    setup()
