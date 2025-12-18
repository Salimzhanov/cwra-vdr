# Contributing to CWRA

Thank you for your interest in contributing to the CWRA (Calibrated Weighted Rank Aggregation) project! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Salimzhanov/cwra-vdr-toolbox.git
   cd cwra-vdr-toolbox
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

This project follows PEP 8 style guidelines. We use:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

Run the following before committing:
```bash
black cwra/
flake8 cwra/
mypy cwra/
```

## Testing

Run the test suite:
```bash
pytest
```

## Documentation

Update documentation in the `docs/` directory and README.md as needed.

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add tests
4. Run the full test suite and linting
5. Commit your changes: `git commit -m "Add your feature"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Create a Pull Request

## Issues

- Use GitHub issues to report bugs or request features
- Provide detailed information including:
  - Python version
  - Operating system
  - Steps to reproduce
  - Expected vs actual behavior

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors.

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.