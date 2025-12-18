# ERNIE Documentation

Source files for contents presented at [https://ernie.readthedocs.io/](https://ernie.readthedocs.io/en/latest/).

## Contributing

You can build the docs as web pages locally.

```bash
cd docs
uv venv
source .venv/bin/activate
uv run mkdocs serve
```

### Markdown Linting

We use `pymarkdown` to lint markdown files. Install the linting tools with:

```bash
uv sync --extra lint
```

Check all markdown files:

```bash
uv run pymarkdown scan source/
```

Auto-fix issues (where supported):

```bash
uv run pymarkdown fix source/
```
