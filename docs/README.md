# ERNIE Documentation

Welcome to the ERNIE documentation.

## Guides

- [ERNIEKit Overview & Installation](erniekit.md)
- [WebUI & CLI Usage](cli_webui_usage.md)
- [PaddleOCR-VL SFT](paddleocr_vl_sft.md)
- [Datasets Guide](datasets.md)
- [Training & Evaluation Arguments](training_eval_args.md)
- [Chat Arguments](chat_args.md)
- [Export Arguments](export_args.md)
- [FP8 Quantization-Aware Training (QAT)](fp8_qat.md)
- [WINT8 Mixed Precision LoRA](wint8mix_lora.md)
- [Unified Checkpoint](unified_checkpoint.md)

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
