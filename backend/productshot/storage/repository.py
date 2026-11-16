"""Repositorio SQLite de sesiones, trabajos, variaciones, decisiones y resúmenes.

Cada transición de estado se ejecuta dentro de `BEGIN IMMEDIATE`, de modo que la
verificación del estado y la escritura son atómicas aunque haya hilos concurrentes
(el ejecutor de trabajos corre en otro hilo). En GCP se sustituirá por Firestore con
las mismas garantías mediante transacciones de documento.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from productshot.domain import (
    Category,
    DECIDABLE_STATES,
    VARIATIONS_PER_BATCH,
    Decision,
    DecisionValue,
    GenerationJob,
    InputImage,
    JobStatus,
    Mask,
    Mode,
    NotFound,
    Session,
    SessionStatus,
    StateConflict,
    Style,
    SummaryRecord,
    Variation,
    decisions_hash,
    iso,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK (mode IN ('intermediate', 'advanced')),
    status TEXT NOT NULL,
    image_id TEXT NOT NULL,
    base_model_revision TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parent_session_id TEXT REFERENCES sessions(id),
    round_index INTEGER NOT NULL DEFAULT 1 CHECK (round_index BETWEEN 1 AND 3),
    round_plan_json TEXT,
    category TEXT
);
-- Una ronda tiene a lo sumo una hija (ADR-007).
CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_parent ON sessions(parent_session_id);
CREATE TABLE IF NOT EXISTS input_images (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id),
    storage_key TEXT NOT NULL,
    original_key TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    original_width INTEGER NOT NULL,
    original_height INTEGER NOT NULL,
    original_format TEXT NOT NULL,
    quality_json TEXT
);
CREATE TABLE IF NOT EXISTS masks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id),
    source TEXT NOT NULL CHECK (source IN ('auto', 'manual')),
    storage_key TEXT NOT NULL,
    preview_key TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    simulated INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    warnings_json TEXT
);
CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id),
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT,
    config_json TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS variations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    job_id TEXT NOT NULL REFERENCES generation_jobs(id),
    position INTEGER NOT NULL CHECK (position BETWEEN 1 AND 5),
    style TEXT NOT NULL,
    variant TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    seed INTEGER NOT NULL,
    model_revision TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    UNIQUE (session_id, position),
    -- Desde las rondas, un estilo aceptado se repite: lo único es (estilo + ajustes).
    UNIQUE (session_id, variant)
);
CREATE TABLE IF NOT EXISTS decisions (
    variation_id TEXT PRIMARY KEY REFERENCES variations(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    value TEXT NOT NULL CHECK (value IN ('accepted', 'rejected')),
    reason TEXT,
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    text TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validated_decisions_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""
SCHEMA_VERSION = 4

# Hito 1 original (versión 0): sin rondas y con UNIQUE (session_id, style).
_MIGRATE_0_TO_2 = """
ALTER TABLE sessions ADD COLUMN parent_session_id TEXT REFERENCES sessions(id);
ALTER TABLE sessions ADD COLUMN round_index INTEGER NOT NULL DEFAULT 1 CHECK (round_index BETWEEN 1 AND 3);
ALTER TABLE sessions ADD COLUMN round_plan_json TEXT;
CREATE TABLE variations_v2 (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    job_id TEXT NOT NULL REFERENCES generation_jobs(id),
    position INTEGER NOT NULL CHECK (position BETWEEN 1 AND 5),
    style TEXT NOT NULL,
    variant TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    seed INTEGER NOT NULL,
    model_revision TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    UNIQUE (session_id, position),
    UNIQUE (session_id, variant)
);
INSERT INTO variations_v2 (id, session_id, job_id, position, style, variant, storage_key, seed,
                           model_revision, prompt_version, params_json)
    SELECT id, session_id, job_id, position, style, style, storage_key, seed,
           model_revision, prompt_version, params_json FROM variations;
DROP TABLE variations;
ALTER TABLE variations_v2 RENAME TO variations;
"""


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


class SqliteRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _migrate(self) -> None:
        """Lleva una base de una versión anterior al esquema actual, paso a paso.

        Sigue el procedimiento de SQLite para cambiar restricciones: tabla nueva, copia,
        borrado y renombre, con las claves foráneas desactivadas y verificadas al final.
        """
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            has_sessions = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'"
            ).fetchone()
            if version >= SCHEMA_VERSION or not has_sessions:
                return
            def cols(table: str) -> set[str]:
                return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("BEGIN IMMEDIATE")
            try:
                if "round_index" not in cols("sessions"):  # versión 0 -> 2: rondas
                    for statement in _MIGRATE_0_TO_2.split(";"):
                        if statement.strip():
                            conn.execute(statement)
                # 2 -> 3: métricas y avisos de los guardrails G12 y G13 (columnas nuevas, sin restricciones).
                # Si una tabla no existe, SCHEMA la crea después ya con la columna.
                if cols("input_images") and "quality_json" not in cols("input_images"):
                    conn.execute("ALTER TABLE input_images ADD COLUMN quality_json TEXT")
                if cols("masks") and "warnings_json" not in cols("masks"):
                    conn.execute("ALTER TABLE masks ADD COLUMN warnings_json TEXT")
                # 3 -> 4: categoría del producto (modo intermedio). Nula en las sesiones anteriores.
                if "category" not in cols("sessions"):
                    conn.execute("ALTER TABLE sessions ADD COLUMN category TEXT")
                broken = conn.execute("PRAGMA foreign_key_check").fetchall()
                if broken:
                    raise sqlite3.IntegrityError(f"La migración dejó claves foráneas rotas: {broken[:3]}")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    # --- infraestructura ---

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Transacción de escritura con bloqueo inmediato."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    @staticmethod
    def _status(conn: sqlite3.Connection, session_id: str) -> SessionStatus:
        row = conn.execute("SELECT status FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise NotFound("La sesión no existe.")
        return SessionStatus(row["status"])

    @staticmethod
    def _set_status(
        conn: sqlite3.Connection, session_id: str, status: SessionStatus, failure_reason: str | None = None
    ) -> None:
        conn.execute(
            "UPDATE sessions SET status = ?, failure_reason = ?, updated_at = ? WHERE id = ?",
            (status.value, failure_reason, iso(utcnow()), session_id),
        )

    # --- sesiones e imágenes ---

    def create_session(self, session: Session, image: InputImage) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO sessions (id, mode, status, image_id, base_model_revision, failure_reason,"
                " created_at, updated_at, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id, session.mode.value, session.status.value, session.image_id,
                    session.base_model_revision, None, iso(session.created_at), iso(session.updated_at),
                    session.category.value if session.category else None,
                ),
            )
            self._insert_image(conn, image)

    @staticmethod
    def _insert_image(conn: sqlite3.Connection, image: InputImage) -> None:
        conn.execute(
            "INSERT INTO input_images (id, session_id, storage_key, original_key, sha256, width, height,"
            " original_width, original_height, original_format, quality_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image.id, image.session_id, image.storage_key, image.original_key, image.sha256,
                image.width, image.height, image.original_width, image.original_height, image.original_format,
                image.quality_json,
            ),
        )

    @staticmethod
    def _insert_mask(conn: sqlite3.Connection, mask: Mask) -> None:
        conn.execute(
            "INSERT INTO masks (id, session_id, source, storage_key, preview_key, sha256, simulated, created_at,"
            " warnings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mask.id, mask.session_id, mask.source, mask.storage_key, mask.preview_key, mask.sha256,
                int(mask.simulated), iso(mask.created_at), mask.warnings_json,
            ),
        )

    def create_child_session(self, child: Session, image: InputImage, mask: Mask, max_rounds: int) -> None:
        """Crea la ronda siguiente, ya en SEGMENTED, con la imagen y la máscara de la madre.

        Todo se verifica en la misma transacción: la madre debe estar SUMMARIZED, no tener
        hija y no haber alcanzado el máximo de rondas. El índice único sobre
        `parent_session_id` impide dos hijas aunque lleguen dos solicitudes a la vez.
        """
        parent_id = child.parent_session_id
        with self._tx() as conn:
            row = conn.execute("SELECT status, round_index FROM sessions WHERE id = ?", (parent_id,)).fetchone()
            if row is None:
                raise NotFound("La sesión no existe.")
            if row["status"] != SessionStatus.SUMMARIZED.value:
                raise StateConflict("Para pedir otra ronda, primero resume las cinco decisiones de esta.")
            if row["round_index"] >= max_rounds:
                raise StateConflict(f"Se alcanzó el máximo de {max_rounds} rondas.")
            existing = conn.execute("SELECT id FROM sessions WHERE parent_session_id = ?", (parent_id,)).fetchone()
            if existing is not None:
                raise StateConflict("Esta ronda ya tiene una ronda siguiente.")
            conn.execute(
                "INSERT INTO sessions (id, mode, status, image_id, base_model_revision, failure_reason,"
                " created_at, updated_at, parent_session_id, round_index, round_plan_json, category)"
                " VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
                (
                    child.id, child.mode.value, child.status.value, child.image_id, child.base_model_revision,
                    iso(child.created_at), iso(child.updated_at), parent_id, child.round_index,
                    child.round_plan_json, child.category.value if child.category else None,
                ),
            )
            self._insert_image(conn, image)
            self._insert_mask(conn, mask)

    def delete_chain(self, session_id: str) -> list[str]:
        """Borra todas las rondas de la cadena a la que pertenece la sesión, en una transacción.

        No borra si alguna ronda está generando: el trabajo escribiría archivos después
        del borrado. Devuelve los IDs borrados (la raíz primero).
        """
        with self._tx() as conn:
            self._status(conn, session_id)  # NotFound si no existe
            root = session_id
            while True:
                row = conn.execute("SELECT parent_session_id FROM sessions WHERE id = ?", (root,)).fetchone()
                if row["parent_session_id"] is None:
                    break
                root = row["parent_session_id"]
            chain = [root]
            while True:
                row = conn.execute("SELECT id FROM sessions WHERE parent_session_id = ?", (chain[-1],)).fetchone()
                if row is None:
                    break
                chain.append(row["id"])
            marks = ",".join("?" * len(chain))
            busy = conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE id IN ({marks}) AND status = ?",
                (*chain, SessionStatus.GENERATING.value),
            ).fetchone()[0]
            if busy:
                raise StateConflict("Hay una generación en curso; espera a que termine para borrar.")
            for table in ("decisions", "summaries", "variations", "generation_jobs", "masks", "input_images"):
                conn.execute(f"DELETE FROM {table} WHERE session_id IN ({marks})", chain)
            # De la hija más nueva a la raíz, por la clave foránea parent_session_id.
            for sid in reversed(chain):
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        return chain

    def child_session_id(self, session_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE parent_session_id = ?", (session_id,)).fetchone()
        return row["id"] if row else None

    def get_session(self, session_id: str) -> Session:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise NotFound("La sesión no existe.")
        return Session(
            id=row["id"],
            mode=Mode(row["mode"]),
            status=SessionStatus(row["status"]),
            image_id=row["image_id"],
            base_model_revision=row["base_model_revision"],
            failure_reason=row["failure_reason"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            parent_session_id=row["parent_session_id"],
            round_index=row["round_index"],
            round_plan_json=row["round_plan_json"],
            category=Category(row["category"]) if row["category"] else None,
        )

    def get_image(self, session_id: str) -> InputImage:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM input_images WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise NotFound("La imagen de la sesión no existe.")
        return InputImage(**dict(row))

    # --- máscara ---

    def replace_mask(self, mask: Mask, allowed: Iterable[SessionStatus]) -> Mask | None:
        """Reemplaza la máscara y deja la sesión en SEGMENTED. Devuelve la anterior."""
        with self._tx() as conn:
            status = self._status(conn, mask.session_id)
            if status not in set(allowed):
                raise StateConflict(f"No se puede cambiar la máscara en estado {status}.")
            parent = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?", (mask.session_id,)
            ).fetchone()["parent_session_id"]
            if parent is not None:
                # La máscara es la de la ronda 1 y sus archivos son de esa sesión: no se toca.
                raise StateConflict("Esta ronda usa la máscara de la ronda 1; no se puede cambiar.")
            previous = self._mask_row(conn, mask.session_id)
            conn.execute("DELETE FROM masks WHERE session_id = ?", (mask.session_id,))
            self._insert_mask(conn, mask)
            self._set_status(conn, mask.session_id, SessionStatus.SEGMENTED)
        return previous

    @staticmethod
    def _mask_row(conn: sqlite3.Connection, session_id: str) -> Mask | None:
        row = conn.execute("SELECT * FROM masks WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["simulated"] = bool(data["simulated"])
        data["created_at"] = _dt(data["created_at"])
        return Mask(**data)

    def get_mask(self, session_id: str) -> Mask | None:
        with self._connect() as conn:
            return self._mask_row(conn, session_id)

    # --- trabajos de generación ---

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> GenerationJob:
        data = dict(row)
        data["status"] = JobStatus(data["status"])
        data["created_at"] = _dt(data["created_at"])
        data["finished_at"] = _dt(data["finished_at"])
        return GenerationJob(**data)

    def get_job(self, session_id: str) -> GenerationJob | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM generation_jobs WHERE session_id = ?", (session_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def start_job(self, job: GenerationJob) -> tuple[GenerationJob, bool]:
        """Crea el único trabajo de la sesión. Devuelve (trabajo, creado).

        Un reintento con la misma `Idempotency-Key` devuelve el trabajo existente.
        Cualquier otro intento sobre una sesión que ya tiene trabajo es un conflicto.
        """
        with self._tx() as conn:
            status = self._status(conn, job.session_id)
            row = conn.execute(
                "SELECT * FROM generation_jobs WHERE session_id = ?", (job.session_id,)
            ).fetchone()
            if row is not None:
                existing = self._job_from_row(row)
                if job.idempotency_key and existing.idempotency_key == job.idempotency_key:
                    return existing, False
                raise StateConflict("La sesión ya tiene una generación. Crea una nueva sesión para generar otra.")
            if status != SessionStatus.SEGMENTED:
                raise StateConflict(f"Solo se puede generar con la máscara lista (estado actual: {status}).")
            conn.execute(
                "INSERT INTO generation_jobs (id, session_id, backend, status, idempotency_key, config_json,"
                " error, created_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL)",
                (
                    job.id, job.session_id, job.backend, JobStatus.GENERATING.value, job.idempotency_key,
                    job.config_json, iso(job.created_at),
                ),
            )
            self._set_status(conn, job.session_id, SessionStatus.GENERATING)
        return job, True

    def complete_job(self, job_id: str, variations: list[Variation]) -> None:
        """Persiste el lote completo de cinco y marca la sesión READY_FOR_REVIEW, todo o nada."""
        if len(variations) != VARIATIONS_PER_BATCH:
            raise ValueError("Un lote completo tiene exactamente cinco variaciones.")
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] != JobStatus.GENERATING.value:
                raise StateConflict("El trabajo no está en curso.")
            session_id = row["session_id"]
            for v in variations:
                conn.execute(
                    "INSERT INTO variations (id, session_id, job_id, position, style, variant, storage_key, seed,"
                    " model_revision, prompt_version, params_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        v.id, session_id, job_id, v.position, v.style.value, v.variant or v.style.value,
                        v.storage_key, v.seed, v.model_revision, v.prompt_version, v.params_json,
                    ),
                )
            conn.execute(
                "UPDATE generation_jobs SET status = ?, finished_at = ? WHERE id = ?",
                (JobStatus.READY_FOR_REVIEW.value, iso(utcnow()), job_id),
            )
            self._set_status(conn, session_id, SessionStatus.READY_FOR_REVIEW)

    def fail_job(self, job_id: str, error: str) -> None:
        with self._tx() as conn:
            row = conn.execute("SELECT session_id, status FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] != JobStatus.GENERATING.value:
                return
            conn.execute(
                "UPDATE generation_jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (JobStatus.FAILED.value, error, iso(utcnow()), job_id),
            )
            self._set_status(conn, row["session_id"], SessionStatus.FAILED, failure_reason=error)

    def fail_interrupted_jobs(self, reason: str) -> int:
        """Al arrancar, ningún trabajo local puede seguir en curso: se marcan FAILED."""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT id, session_id FROM generation_jobs WHERE status = ?", (JobStatus.GENERATING.value,)
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE generation_jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                    (JobStatus.FAILED.value, reason, iso(utcnow()), row["id"]),
                )
                self._set_status(conn, row["session_id"], SessionStatus.FAILED, failure_reason=reason)
        return len(rows)

    # --- variaciones y decisiones ---

    @staticmethod
    def _variations(conn: sqlite3.Connection, session_id: str) -> list[Variation]:
        rows = conn.execute(
            "SELECT v.*, d.value AS d_value, d.reason AS d_reason, d.decided_at AS d_decided_at"
            " FROM variations v LEFT JOIN decisions d ON d.variation_id = v.id"
            " WHERE v.session_id = ? ORDER BY v.position",
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            decision = None
            if r["d_value"] is not None:
                decision = Decision(
                    variation_id=r["id"],
                    value=DecisionValue(r["d_value"]),
                    reason=r["d_reason"],
                    decided_at=_dt(r["d_decided_at"]),
                )
            result.append(
                Variation(
                    id=r["id"], session_id=r["session_id"], job_id=r["job_id"], position=r["position"],
                    style=Style(r["style"]), storage_key=r["storage_key"], seed=r["seed"],
                    model_revision=r["model_revision"], prompt_version=r["prompt_version"],
                    params_json=r["params_json"], decision=decision, variant=r["variant"],
                )
            )
        return result

    def list_variations(self, session_id: str) -> list[Variation]:
        with self._connect() as conn:
            return self._variations(conn, session_id)

    def upsert_decision(
        self, session_id: str, variation_id: str, value: DecisionValue, reason: str | None
    ) -> Decision:
        with self._tx() as conn:
            status = self._status(conn, session_id)
            if status == SessionStatus.SUMMARIZED:
                raise StateConflict("La sesión ya fue resumida; sus decisiones están congeladas.")
            row = conn.execute(
                "SELECT id FROM variations WHERE id = ? AND session_id = ?", (variation_id, session_id)
            ).fetchone()
            if row is None:
                raise NotFound("La variación no existe en esta sesión.")
            if status not in DECIDABLE_STATES:
                raise StateConflict(f"No se aceptan decisiones en estado {status}.")
            decided_at = utcnow()
            conn.execute(
                "INSERT INTO decisions (variation_id, session_id, value, reason, decided_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(variation_id) DO UPDATE SET value = excluded.value, reason = excluded.reason,"
                " decided_at = excluded.decided_at",
                (variation_id, session_id, value.value, reason, iso(decided_at)),
            )
            count = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            if count == VARIATIONS_PER_BATCH and status == SessionStatus.READY_FOR_REVIEW:
                self._set_status(conn, session_id, SessionStatus.REVIEWED)
        return Decision(variation_id=variation_id, value=value, reason=reason, decided_at=decided_at)

    # --- resumen ---

    def save_summary(self, record: SummaryRecord) -> SummaryRecord:
        """Guarda el resumen solo si las decisiones no cambiaron desde que se calculó.

        Devuelve el resumen persistido. Si una solicitud concurrente ya resumió las mismas
        decisiones, devuelve ese resumen en lugar de fallar (POST idempotente).
        """
        with self._tx() as conn:
            status = self._status(conn, record.session_id)
            if status == SessionStatus.SUMMARIZED:
                existing = self._summary_row(conn, record.session_id)
                if existing is not None and existing.validated_decisions_hash == record.validated_decisions_hash:
                    return existing
            if status != SessionStatus.REVIEWED:
                raise StateConflict(f"No se puede resumir en estado {status}.")
            current = decisions_hash(self._variations(conn, record.session_id))
            if current != record.validated_decisions_hash:
                raise StateConflict("Las decisiones cambiaron mientras se generaba el resumen; reinténtalo.")
            conn.execute(
                "INSERT INTO summaries (session_id, text, payload_json, generated_by, validation_status,"
                " validated_decisions_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.session_id, record.text, record.payload_json, record.generated_by,
                    record.validation_status, record.validated_decisions_hash, iso(record.created_at),
                ),
            )
            self._set_status(conn, record.session_id, SessionStatus.SUMMARIZED)
        return record

    @staticmethod
    def _summary_row(conn: sqlite3.Connection, session_id: str) -> SummaryRecord | None:
        row = conn.execute("SELECT * FROM summaries WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["created_at"] = _dt(data["created_at"])
        return SummaryRecord(**data)

    def get_summary(self, session_id: str) -> SummaryRecord | None:
        with self._connect() as conn:
            return self._summary_row(conn, session_id)

    # --- control de acceso a artefactos ---

    def session_artifact_keys(self, session_id: str) -> set[str]:
        """Claves de artefactos que la base de datos asocia a la sesión (y solo esas se sirven)."""
        with self._connect() as conn:
            keys: set[str] = set()
            img = conn.execute(
                "SELECT storage_key, original_key FROM input_images WHERE session_id = ?", (session_id,)
            ).fetchone()
            if img:
                keys.update((img["storage_key"], img["original_key"]))
            mask = conn.execute(
                "SELECT storage_key, preview_key FROM masks WHERE session_id = ?", (session_id,)
            ).fetchone()
            if mask:
                keys.update((mask["storage_key"], mask["preview_key"]))
            for r in conn.execute("SELECT storage_key FROM variations WHERE session_id = ?", (session_id,)):
                keys.add(r["storage_key"])
        return keys
