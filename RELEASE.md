# Release Checklist

This document outlines the steps to release a new version of CWRA to GitHub and PyPI.

## Pre-Release Checklist

- [ ] Update version numbers in:
  - `cwra/__init__.py`
  - `setup.py`
  - `pyproject.toml`
  - `CHANGELOG.md`

- [ ] Update CHANGELOG.md with release notes

- [ ] Run full test suite:
  ```bash
  python -m pytest tests/ -v --cov=cwra
  ```

- [ ] Test installation:
  ```bash
  pip install -e .
  python -c "import cwra; print('Import successful')"
  ```

- [ ] Test examples:
  ```bash
  python examples/basic_example.py
  ```

- [ ] Update documentation if needed

- [ ] Commit all changes:
  ```bash
  git add .
  git commit -m "Release version X.Y.Z"
  git tag vX.Y.Z
  git push origin main --tags
  ```

## GitHub Release

1. Go to [GitHub Releases](https://github.com/yourusername/cwra-vdr-benchmark/releases)
2. Click "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `CWRA v1.0.0 - Initial Public Release`
5. Release notes: Copy from CHANGELOG.md
6. Attach any additional files if needed
7. Publish release

## PyPI Release

1. Build distribution:
   ```bash
   python -m pip install --upgrade build
   python -m build
   ```

2. Upload to PyPI:
   ```bash
   python -m pip install --upgrade twine
   twine upload dist/*
   ```

3. Verify on PyPI: https://pypi.org/project/cwra-vdr/

## Post-Release

- [ ] Update README badges if needed
- [ ] Announce release on relevant forums/channels
- [ ] Monitor for issues and fix promptly
- [ ] Plan next release cycle

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## Release Notes Template

```
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security-related changes
```