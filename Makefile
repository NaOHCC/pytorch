# This makefile does nothing but delegating the actual building to cmake.
PYTHON = python3
PIP = pip3

all:
	@mkdir -p build && cd build && cmake .. $(shell $(PYTHON) ./scripts/get_python_cmake_flags.py) && $(MAKE)

local:
	@./scripts/build_local.sh

android:
	@./scripts/build_android.sh

ios:
	@./scripts/build_ios.sh

clean: # This will remove ALL build folders.
	python setup.py clean

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"

setup_lint:
	$(PIP) install lintrunner
	lintrunner init

lint:
	lintrunner

quicklint:
	lintrunner

triton:
	$(PIP) uninstall -y triton
	@./scripts/install_triton_wheel.sh

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export USE_GLOO=0
export USE_GLOG=1
export USE_NCCL=1
export USE_CUDNN=1
export CUDNN_LIBRARY=/opt/conda/lib
export CUDNN_INCLUDE_DIR=/opt/conda/include
export DEBUG=1
export LD_LIBRARY_PATH=/opt/conda/lib
export ORIGIN=/opt/conda/lib
build:
	python setup.py build

rebuild:
	python setup.py rebuild

cmake:
	python setup.py build --cmake-only

env:
	env
.PHONY: build cmake env rebuild