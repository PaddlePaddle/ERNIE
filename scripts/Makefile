# Makefile for ERNIEKit
#
# 	GitHub: https://github.com/PaddlePaddle/ERNIE
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 
.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi	


