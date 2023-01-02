from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ModelsResponse(_message.Message):
    __slots__ = ["models"]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: str
    def __init__(self, models: _Optional[str] = ...) -> None: ...

class PredictionRequest(_message.Message):
    __slots__ = ["input", "modelName"]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    MODELNAME_FIELD_NUMBER: _ClassVar[int]
    input: bytes
    modelName: str
    def __init__(self, modelName: _Optional[str] = ..., input: _Optional[bytes] = ...) -> None: ...

class PredictionResponse(_message.Message):
    __slots__ = ["error", "prediction"]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    error: str
    prediction: bytes
    def __init__(self, prediction: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...
