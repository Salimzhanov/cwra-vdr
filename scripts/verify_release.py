#!/usr/bin/env python3
"""
Release Verification Script

This script verifies that the CWRA package is ready for public release.
Run this before creating a GitHub release.
"""

import os
import sys
import subprocess
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_package_import():
    """Check if the package can be imported."""
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))

        # Try importing the package
        import cwra
        print(f"✓ Package import successful: cwra v{cwra.__version__}")
        print(f"  Author: {cwra.__author__}")
        return True
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        return False

def check_command_line_tool():
    """Check if the command-line tool works."""
    try:
        # Test the command-line interface
        result = subprocess.run([
            sys.executable, "-m", "cwra", "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode == 0:
            print("✓ Command-line tool works")
            return True
        else:
            print(f"✗ Command-line tool failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Command-line tool error: {e}")
        return False

def check_example_scripts():
    """Check if example scripts can be imported (syntax check)."""
    examples_dir = "examples"
    success = True

    for filename in os.listdir(examples_dir):
        if filename.endswith(".py"):
            filepath = os.path.join(examples_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filepath, 'exec')
                print(f"✓ Example script syntax: {filename}")
            except SyntaxError as e:
                print(f"✗ Example script syntax error in {filename}: {e}")
                success = False

    return success

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("CWRA Release Verification")
    print("=" * 60)

    all_passed = True

    # Check required files
    required_files = [
        ("README.md", "README file"),
        ("LICENSE", "License file"),
        ("requirements.txt", "Requirements file"),
        ("setup.py", "Setup script"),
        ("pyproject.toml", "Modern packaging config"),
        ("cwra/__init__.py", "Package init file"),
        ("cwra/cwra.py", "Main module"),
        ("tests/test_cwra.py", "Test file"),
        ("data/labeled_raw_modalities.csv", "Example data"),
        (".github/workflows/ci.yml", "CI workflow"),
    ]

    print("\n1. Checking required files:")
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_passed = False

    # Check package import
    print("\n2. Checking package import:")
    if not check_package_import():
        all_passed = False

    # Check command-line tool
    print("\n3. Checking command-line tool:")
    if not check_command_line_tool():
        all_passed = False

    # Check example scripts
    print("\n4. Checking example scripts:")
    if not check_example_scripts():
        all_passed = False

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready for release!")
        print("\nNext steps:")
        print("1. Commit all changes")
        print("2. Create git tag: git tag v1.0.0")
        print("3. Push to GitHub: git push origin main --tags")
        print("4. Create GitHub release")
        print("5. Build and upload to PyPI: python -m build && twine upload dist/*")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before release")
        sys.exit(1)

if __name__ == "__main__":
    main()