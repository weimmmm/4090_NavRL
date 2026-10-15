"""Run TensorBoard with its legacy protobuf JSON keyword supported.

The adaptation only affects this dashboard process. Training and validation
continue to use the installed protobuf package without any changes.
"""

from __future__ import annotations

import functools
import inspect

from google.protobuf import json_format


def _compatible_converter(converter):
    parameters = inspect.signature(converter).parameters
    if "including_default_value_fields" in parameters:
        return converter
    if "always_print_fields_with_no_presence" not in parameters:
        raise RuntimeError("Unsupported protobuf JSON converter signature")

    @functools.wraps(converter)
    def convert(*args, **kwargs):
        if "including_default_value_fields" in kwargs:
            legacy = kwargs.pop("including_default_value_fields")
            kwargs.setdefault("always_print_fields_with_no_presence", legacy)
        return converter(*args, **kwargs)

    return convert


def main():
    json_format.MessageToJson = _compatible_converter(json_format.MessageToJson)
    json_format.MessageToDict = _compatible_converter(json_format.MessageToDict)
    # Import the dashboard after adapting the functions, so plugins also see
    # the compatible converters if they import them directly.
    from tensorboard.main import run_main

    run_main()


if __name__ == "__main__":
    main()
