UNAME := $(shell uname)

deps-macos:
	@if [ -e "/tmp/libtorch.zip" ]; then \
		echo "LibTorch for macOS already exists"; \
	else \
		echo "Downloading and Installing LibTorch for macOS"; \
		curl -o /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.2.zip; \
	fi
	@unzip /tmp/libtorch.zip -d /tmp
	@sudo mv /tmp/libtorch /usr/local/libtorch

deps-linux:
	@if [ -e "/tmp/libtorch.zip" ]; then \
		echo "LibTorch for Linux already exists"; \
	else \
		echo "Downloading and Installing LibTorch for Linux"; \
		curl -o /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip \
	fi
	@unzip /tmp/libtorch.zip -d /tmp
	@sudo mv /tmp/libtorch /usr/local/libtorch

deps:
	@if [ "$(UNAME)" = "Darwin" ]; then \
		$(MAKE) deps-macos; \
	else \
		$(MAKE) deps-linux; \
	fi

	@echo "Downloaing Go Dependencies"
	@go mod download

run:
	@if [ "$(UNAME)" = "Darwin" ]; then \
		export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH}" && \
			export DYLD_LIBRARY_PATH="/usr/local/libtorch/lib:${DYLD_LIBRARY_PATH}" && \
			export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}" && \
			export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}" && go run main.go; \
	else \
		export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH}" && \
			export LD_LIBRARY_PATH="/usr/local/libtorch/lib:${LD_LIBRARY_PATH}" && \
			export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}" && \
			export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}" go run main.go; \
	fi
