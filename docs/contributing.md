# Contributing

MediPixel is open to contributions. This page explains how to get set up,
what the standards are, and how to submit changes.

---

## Before you start

For anything beyond a typo fix, **open an issue first**. This avoids you
spending time on something that won't be merged. Use one of the templates:

- [Bug report](https://github.com/BasselShaheen06/MediPixel/issues/new?template=bug_report.md)
- [Feature request](https://github.com/BasselShaheen06/MediPixel/issues/new?template=feature_request.md)

---

## Development setup

```bash
git clone https://github.com/BasselShaheen06/MediPixel.git
cd MediPixel

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install mkdocs-material     # for docs only

python main.py                  # verify setup
```

---

## Project structure

```
MediPixel/
├── main.py                     # entry point only — 10 lines
├── medipixel/
│   ├── core/                   # pure processing logic — no UI
│   │   ├── noise.py
│   │   ├── denoiser.py
│   │   └── contrast.py
│   └── ui/                     # Qt + pyqtgraph UI layer
│       ├── canvas.py
│       └── main_window.py
└── docs/                       # MkDocs source
```

**Hard boundaries — do not cross these:**

- `core/` modules must never import `PyQt5`, `pyqtgraph`, or `matplotlib`
- `core/` functions take a numpy array, return a numpy array — no side effects
- `ui/main_window.py` owns the processing pipeline — `core/` functions are called from there, not from each other

If you need to add a new processing operation, add it to the appropriate `core/` file and wire it into `_process()` in `main_window.py`. Follow the existing pattern exactly.

---

## Code standards

### Formatting

Use `black` for formatting. Line length 100.

```bash
pip install black
black medipixel/
```

### Type hints

All new functions in `core/` must have type hints:

```python
# correct
def my_filter(image: np.ndarray, strength: float = 1.0) -> np.ndarray:

# wrong
def my_filter(image, strength=1.0):
```

### Docstrings

Use Google-style docstrings:

```python
def my_filter(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Short description of what this does.

    Args:
        image:    2-D uint8 numpy array.
        strength: Control parameter. Range [0, 1]. Higher = stronger effect.

    Returns:
        Processed image as uint8, same shape as input.
    """
```

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Wiener denoising filter
fix: correct ROI coordinate mapping after zoom
docs: add bilateral filter parameter guide
refactor: extract frequency filter into standalone function
test: add unit test for CLAHE clip limit boundary
```

---

## Submitting a pull request

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature-name`
3. Make your changes — keep PRs focused on one thing
4. Run `black medipixel/` before committing
5. Push and open a PR against `main`
6. Fill in the PR description — what changed and why

**PRs that will be merged quickly:**

- Bug fixes with a clear description of the root cause
- New processing operations following the existing `core/` pattern
- Documentation improvements
- Performance improvements with before/after measurements

**PRs that need discussion first** (open an issue before starting):

- Changes to the UI layout
- New dependencies
- Anything that touches the ROI workflow or SNR/CNR calculation

---

## Adding a new processing operation

1. Add the function to the appropriate `core/` file (`noise.py`, `denoiser.py`, or `contrast.py`)
2. Follow the pure-function pattern: `np.ndarray → np.ndarray`, no side effects
3. Add a docstring with Args and Returns
4. Wire it into `_process()` in `ui/main_window.py`
5. Add a dropdown option to the relevant `_combo()` in `_build_sidebar()`
6. Document it in `docs/api/`

---

## Running the documentation locally

```bash
pip install mkdocs-material
mkdocs serve
# open http://127.0.0.1:8000
```