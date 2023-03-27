deps:
	@echo "Downloading and Installing LibTorch"
	@wget -O /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
	@unzip /tmp/libtorch.zip -d /tmp
	@sudo mv /tmp/libtorch /usr/local/libtorch
	@export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH}"
	@export LD_LIBRARY_PATH="/usr/local/libtorch/lib:${LD_LIBRARY_PATH}"
	@export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}"
	@export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}"
	@echo "Downloaing Go Dependencies"
	@go mod download

run:
	@go run main.go
