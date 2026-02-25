# Contributing

## Setup

```bash
git clone https://github.com/Salimzhanov/cwra-vdr.git
cd cwra-vdr
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Code style

We use `ruff` for linting. Run before committing:

```bash
ruff check cwra/ cwra.py pu_conformal.py run_cwra.py
```

## Tests

```bash
pytest tests/ -v
```

## Pull requests

1. Fork & branch (`git checkout -b fix/description`)
2. Make changes, add tests if applicable
3. `pytest tests/ -v` must pass
4. Open a PR against `master`

## Bugs

File an issue with your Python version, OS, and steps to reproduce.

## License

Contributions are licensed under the same MIT license as the project.