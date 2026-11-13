"""Almacenamiento de artefactos (imágenes y máscaras).

Las claves son relativas y se validan con una expresión estricta: nunca se concatena
input del usuario a una ruta. En GCP se sustituirá por un adaptador de Cloud Storage.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

# <uuid>/<nombre>.png, solo minúsculas, dígitos, guion y guion bajo.
_SESSION_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/[a-z0-9_\-]{1,64}\.(png|jpg)$")


class InvalidArtifactKey(ValueError):
    pass


def validate_key(key: str) -> str:
    if not _KEY_RE.fullmatch(key):
        raise InvalidArtifactKey(f"Clave de artefacto inválida: {key!r}")
    return key


class ArtifactStore(Protocol):
    def save(self, data: bytes, key: str) -> str: ...
    def load(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def delete_session_dir(self, session_id: str) -> None: ...


class LocalArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_key(key)
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise InvalidArtifactKey(f"Clave fuera del almacenamiento: {key!r}")
        return path

    def save(self, data: bytes, key: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atómica: un lector nunca ve un archivo a medio escribir.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return key

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_session_dir(self, session_id: str) -> None:
        """Borra la carpeta de una sesión con todo lo que tenga, incluidos temporales huérfanos."""
        if not _SESSION_RE.fullmatch(session_id):
            raise InvalidArtifactKey(f"ID de sesión inválido: {session_id!r}")
        path = (self.root / session_id).resolve()
        if path.parent != self.root:
            raise InvalidArtifactKey(f"Carpeta fuera del almacenamiento: {session_id!r}")
        shutil.rmtree(path, ignore_errors=True)
