"""Durable originals for temporary v2 F0/CD changes; no CAN or ROS operations."""

import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile

from .mit_protocol import parse_f0


class SettingsJournal:
    def __init__(self, path, device_ids):
        self.path = Path(path).expanduser()
        self.device_ids = device_ids
        self.document = None

    def _sync_parent(self):
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def claim(self, originals, pending):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = dict(schema="trackstand_v2_mit_settings_v1", device_ids=self.device_ids,
                        originals=dict(originals), pending=dict(pending),
                        steering_B1="Persistent straight origin; not restored")
        # Exclusive ownership also prevents two v2 workers sharing this journal.
        with self.path.open("x") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sync_parent()
        self.document = document  # Never claim an existing or incompletely written journal.

    def load(self):
        document = json.loads(self.path.read_text())
        if document.get("schema") != "trackstand_v2_mit_settings_v1" or document.get("device_ids") != self.device_ids:
            raise ValueError("pending settings journal belongs to a different motor configuration")
        for collection in ("originals", "pending"):
            values = document.get(collection)
            if not isinstance(values, dict):
                raise ValueError("invalid settings journal")
            for key, value in values.items():
                if key not in {f"{name}:{op}" for name in ("steering", "drive") for op in ("F0", "CD")}:
                    raise ValueError("unexpected restore operation in settings journal")
                if not isinstance(value, str):
                    raise ValueError("invalid restore payload in settings journal")
                payload = bytes.fromhex(value)
                if (not payload or payload[0] != int(key.split(":")[1], 16)
                        or len(payload) != (7 if payload[0] == 0xF0 else 5)):
                    raise ValueError("invalid restore payload in settings journal")
                if collection == "pending" and document["originals"].get(key) != value:
                    raise ValueError("restore payload differs from recorded original")
                if payload[0] == 0xF0:
                    parse_f0(payload)
                elif payload[1] not in (0, 1):
                    raise ValueError("invalid original watchdog setting")
        self.document = document

    @property
    def pending(self):
        return {} if self.document is None else self.document["pending"]

    def restored(self, key):
        document = deepcopy(self.document)
        document["pending"].pop(key)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                             prefix=self.path.name, delete=False) as stream:
                temp = Path(stream.name)
                json.dump(document, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
            self._sync_parent()
            self.document = document
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def finish(self):
        if self.pending:
            raise RuntimeError("motor settings restoration incomplete")
        if self.document is not None:
            self.path.unlink()
            self._sync_parent()
            self.document = None
