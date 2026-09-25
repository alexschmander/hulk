"""Decoder for ROS-Z CDR single images and stereo image pairs in MCAP."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path

from mcap.exceptions import McapError
from mcap.records import Channel, Footer, Message, Schema
from mcap.stream_reader import StreamReader

from .common import Record


class Cdr:
    """Only the explicitly validated image schemas below, XCDR1 alignment."""

    def __init__(self, data: bytes) -> None:
        if data[:4] not in (b"\x00\x01\x00\x00", b"\x00\x00\x00\x00"):
            message = "Unsupported CDR encapsulation"
            raise ValueError(message)
        self.data = memoryview(data)
        self.pos = 4
        self.endian = "<" if data[1] == 1 else ">"

    def scalar(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        self.pos += (-(self.pos - 4)) % size
        value = struct.unpack_from(self.endian + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def bytes(self, count: int) -> memoryview:
        end = self.pos + count
        if end > len(self.data):
            message = "Truncated CDR payload"
            raise ValueError(message)
        value = self.data[self.pos : end]
        self.pos = end
        return value

    def string(self) -> str:
        value = self.bytes(self.scalar("I"))
        if not value or value[-1] != 0:
            message = "Invalid CDR string"
            raise ValueError(message)
        return bytes(value[:-1]).decode("utf-8")

    def image(self) -> Record:
        sec, nsec = self.scalar("i"), self.scalar("I")
        frame_id = self.string()
        height, width = self.scalar("I"), self.scalar("I")
        encoding, bigendian = self.string(), self.scalar("B")
        step = self.scalar("I")
        data = self.bytes(self.scalar("I"))
        if encoding != "nv12" or width % 2 or height % 2 or step < width:
            message = (
                f"Unsupported image layout: {encoding} "
                f"{width}x{height} stride {step}"
            )
            raise ValueError(message)
        if len(data) != step * height * 3 // 2:
            message = "NV12 payload size does not match stride and dimensions"
            raise ValueError(message)
        return {
            "width": width,
            "height": height,
            "step": step,
            "encoding": encoding,
            "frame_id": frame_id,
            "capture_ns": sec * 10**9 + nsec,
            "valid_timestamp": sec >= 0 and 0 <= nsec < 10**9,
            "bigendian": bigendian,
            "data": data,
        }


def validate_schema(schema: Schema, *, stereo: bool) -> None:
    if schema.encoding != "ros-z-schema-json":
        message = f"Unsupported schema encoding {schema.encoding}"
        raise ValueError(message)
    definitions = json.loads(schema.data)["definitions"]
    image_def = definitions["ros2::sensor_msgs::image::Image"]["Struct"][
        "fields"
    ]
    expected = [
        "header",
        "height",
        "width",
        "encoding",
        "is_bigendian",
        "step",
        "data",
    ]
    if [f["name"] for f in image_def] != expected:
        message = "Unexpected Image schema layout"
        raise ValueError(message)
    if stereo:
        fields = definitions["types::stereo_image_pair::StereoImagePair"][
            "Struct"
        ]["fields"]
        if [f["name"] for f in fields] != ["frame_identifier", "left", "right"]:
            message = "Unexpected StereoImagePair layout"
            raise ValueError(message)


def image_messages(
    path: Path, audit: Record, topic: str, *, stereo: bool
) -> Iterator[tuple[Record, Record]]:
    """Yield complete images and retain an incomplete-tail audit."""
    schemas, channels = {}, {}
    ordinal = 0
    audit.update(
        footer_seen=False,
        stream_error=None,
        image_channels=[],
        complete_images=0,
    )
    with path.open("rb") as file:
        iterator = iter(StreamReader(file).records)
        while True:
            try:
                record = next(iterator)
            except StopIteration:
                break
            except (McapError, EOFError, struct.error) as error:
                audit["stream_error"] = f"{type(error).__name__}: {error}"
                audit["bytes_read_at_error"] = file.tell()
                break
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
                if record.topic == topic:
                    validate_channel(
                        record, schemas[record.schema_id], stereo=stereo
                    )
                    audit["image_channels"].append(record.topic)
            elif isinstance(record, Footer):
                audit["footer_seen"] = True
            elif isinstance(record, Message):
                channel = channels[record.channel_id]
                if channel.topic != topic:
                    continue
                left, frame_identifier = decode_image_message(
                    record.data, stereo=stereo
                )
                ordinal += 1
                audit["complete_images"] = ordinal
                yield (
                    left,
                    {
                        "image_ordinal": ordinal,
                        "frame_identifier": frame_identifier,
                        "log_ns": record.log_time,
                        "channel": channel.topic,
                    },
                )


def validate_channel(channel: Channel, schema: Schema, *, stereo: bool) -> None:
    if channel.message_encoding != "ros-z-cdr":
        message = f"Unsupported message encoding {channel.message_encoding}"
        raise ValueError(message)
    validate_schema(schema, stereo=stereo)


def decode_image_message(
    data: bytes, *, stereo: bool
) -> tuple[Record, int | None]:
    cdr = Cdr(data)
    frame_identifier = cdr.scalar("I") if stereo else None
    left = cdr.image()
    if stereo:
        cdr.image()  # Validate the whole pair; only export left.
    if cdr.pos != len(data):
        message = "Unexpected trailing bytes in image message"
        raise ValueError(message)
    return left, frame_identifier
