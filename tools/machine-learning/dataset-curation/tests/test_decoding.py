import struct

import numpy as np
import pytest
from dataset_curation.mcap_io import decode_image_message
from dataset_curation.quality import rgb709


def encoded_image(endian: str) -> bytes:
    data = bytearray(
        b"\x00\x01\x00\x00" if endian == "<" else b"\x00\x00\x00\x00"
    )

    def scalar(fmt: str, value: int) -> None:
        data.extend(b"\x00" * (-(len(data) - 4) % struct.calcsize(fmt)))
        data.extend(struct.pack(endian + fmt, value))

    def string(value: bytes) -> None:
        scalar("I", len(value) + 1)
        data.extend(value + b"\x00")

    scalar("i", 12)
    scalar("I", 345)
    string(b"x")
    scalar("I", 2)
    scalar("I", 2)
    string(b"nv12")
    scalar("B", 0)
    scalar("I", 4)
    pixels = bytes([16, 235, 99, 99] * 2 + [128, 128, 99, 99])
    scalar("I", len(pixels))
    data.extend(pixels)
    return bytes(data)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_cdr_padding_endianness_and_nv12_stride(endian: str) -> None:
    image, frame = decode_image_message(encoded_image(endian), stereo=False)
    assert frame is None
    assert image["capture_ns"] == 12_000_000_345
    assert image["frame_id"] == "x"
    rgb = rgb709(image)
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[:, 0], np.zeros((2, 3)))
    np.testing.assert_array_equal(rgb[:, 1], np.full((2, 3), 255))


def test_truncated_and_trailing_image_payloads_are_rejected() -> None:
    with pytest.raises(ValueError, match="Truncated"):
        decode_image_message(encoded_image("<")[:-1], stereo=False)
    with pytest.raises(ValueError, match="trailing"):
        decode_image_message(encoded_image("<") + b"bad", stereo=False)
