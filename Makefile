# Makefile For ERNIE
# GitHb: https://github.com/PaddlePaddle/ERNIE


.PHONY: all
all : lint gpu_ci_test xpu_ci_test
check_dirs := cookbook data_processor ernie erniekit examples tools tests requirements

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # #
.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python tools/codestyle/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi

# # # # # # # # # # # # # # # Install Requirements Block # # # # # # # # # # # # # # #
.PHONY: install
install:
	pip uninstall paddlepaddle-gpu -y
	pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.6-Cudnn9.5-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
	pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/FastDeploy_ActionCE/SM80_90/release/2.2/latest/fastdeploy_gpu-0.0.0-py3-none-any.whl
	pip install -r requirements/gpu/requirements.txt
	pip install pytest
	pip install allure-pytest
	pip install -e.

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # #
.PHONY: gpu_ci_test
gpu_ci_test:
	PYTHONPATH=$(shell pwd) pytest -s -v --alluredir=result tests/gpu/

.PHONY: xpu_ci_test
xpu_ci_test:
	PYTHONPATH=$(shell pwd) pytest -s -v --alluredir=result tests/xpu/

.PHONY: npu_ci_test
npu_ci_test:
	PYTHONPATH=$(shell pwd) pytest -s -v --alluredir=result tests/npu/
