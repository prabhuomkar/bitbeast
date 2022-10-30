"""TorchLego Input Stage"""
import io
import logging
from typing import Any

from PIL import Image

def read_file(request):
    """Read file"""
    file = request.files.get("input")
    try:
        img = Image.open(io.BytesIO(file.body))
        return img
    except Exception as exp: # pylint: disable=broad-except
        logging.error("error reading input file: %s", str(exp))
    return None

CUSTOM_MAPPER = {
    "file": read_file
}

def derive_input(input_type: str) -> Any:
    """Derive input stage"""
    # default input stage
    return CUSTOM_MAPPER.get(input_type, None)
