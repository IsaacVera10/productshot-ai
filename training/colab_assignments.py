"""Lista o libera asignaciones de VM de Colab directamente en el servidor.

Respaldo de `run_colab.py` para el caso que `colab stop -s NOMBRE` no cubre: un VM asignado
en el servidor pero sin nombre local (por ejemplo, si `colab new` se corta después de que
el servidor asignó la GPU). Usa la API interna del Colab CLI 0.8.0, así que corre con el
mismo intérprete del CLI (la primera línea del ejecutable `colab`):

    <python del CLI> training/colab_assignments.py list
    <python del CLI> training/colab_assignments.py unassign ENDPOINT [ENDPOINT ...]

`list` imprime un JSON con `endpoint` y `accelerator` de cada asignación.
"""

from __future__ import annotations

import json
import sys


def _state():
    from colab_cli.auth import AuthProvider
    from colab_cli.common import state

    state.auth_provider = AuthProvider.OAUTH2
    return state


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("list", "unassign"):
        print(__doc__, file=sys.stderr)
        return 2
    client = _state().client
    if argv[0] == "list":
        rows = [{"endpoint": a.endpoint, "accelerator": str(getattr(a.accelerator, "value", a.accelerator))}
                for a in client.list_assignments()]
        print(json.dumps(rows))
        return 0
    failed = 0
    for endpoint in argv[1:]:
        try:
            client.unassign(endpoint)
            print(f"liberado {endpoint}")
        except Exception as exc:  # noqa: BLE001 - se informa y se sigue con el resto
            failed += 1
            print(f"no se pudo liberar {endpoint}: {exc!r}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
