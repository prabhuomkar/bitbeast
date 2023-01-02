run:
	@python3 -m torchlego.main

deps:
	@pip install -r requirements.txt

proto:
	@python3 -m grpc_tools.protoc -I./ --python_out=./ --pyi_out=./ --grpc_python_out=./ ./torchlego/proto/*.proto

lint:
	@pylint *

build:
	@docker build . -t torchlego
