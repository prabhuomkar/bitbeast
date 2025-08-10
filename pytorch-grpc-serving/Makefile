deps:
	@pip install -r requirements.txt

proto:
	@python3 -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/inference.proto

lint:
	@pylint *.py

build:
	@docker build --no-cache --platform linux/amd64 . -t pytorch-serving
