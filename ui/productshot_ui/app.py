"""UI Gradio del Hito 1.

La UI no accede a la base de datos: todo pasa por la API, igual que pasará en GCP.
Los resultados simulados se rotulan como tales y el resumen se muestra como texto plano
(los motivos del usuario nunca se interpretan como Markdown o HTML).
"""

from __future__ import annotations

import io
import time
from urllib.parse import urlsplit

import gradio as gr
import httpx
from PIL import Image

STYLE_LABELS = {
    "studio": "Estudio",
    "natural": "Natural",
    "warm": "Cálido",
    "lifestyle": "Lifestyle",
    "minimal": "Minimalista",
}
MODE_CHOICES = [
    ("Intermedio: inpainting preentrenado", "intermediate"),
    ("Avanzado: inpainting + LoRA", "advanced"),
]
DECISION_CHOICES = [("Aceptar", "accepted"), ("Rechazar", "rejected")]
POLL_TIMEOUT_S = 180
N = 5

MOCK_NOTICE = (
    "> **Hito 1: modo simulación.** La máscara, las cinco imágenes y el resumen se producen con "
    "componentes de prueba (mock). No hay generación con IA ni agente LLM todavía."
)


class ApiError(Exception):
    pass


class Api:
    def __init__(self, api_base: str):
        parts = urlsplit(api_base)
        self.root = f"{parts.scheme}://{parts.netloc}"
        self.prefix = parts.path.rstrip("/")
        self.http = httpx.Client(base_url=self.root, timeout=60)

    def _call(self, method: str, path: str, **kwargs) -> dict:
        try:
            r = self.http.request(method, self.prefix + path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"No se pudo contactar a la API: {exc}") from exc
        if r.status_code >= 400:
            try:
                msg = r.json()["error"]["message"]
            except Exception:
                msg = r.text[:200]
            raise ApiError(f"{msg} (HTTP {r.status_code})")
        return r.json()

    def image(self, url: str) -> Image.Image:
        try:
            r = self.http.get(url)
        except httpx.HTTPError as exc:
            raise ApiError(f"No se pudo descargar la imagen: {exc}") from exc
        if r.status_code >= 400:
            raise ApiError(f"No se pudo descargar la imagen (HTTP {r.status_code})")
        return Image.open(io.BytesIO(r.content))

    def create_session(self, path: str, mode: str) -> dict:
        ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as fh:
            return self._call("POST", "/sessions", files={"image": ("upload", fh, ctype)}, data={"mode": mode})

    def segment(self, sid: str) -> dict:
        return self._call("POST", f"/sessions/{sid}/segment")

    def generate(self, sid: str) -> dict:
        # Llave estable por sesión: un doble clic no crea un segundo trabajo.
        return self._call("POST", f"/sessions/{sid}/generate", headers={"Idempotency-Key": f"ui-{sid}"})

    def session(self, sid: str) -> dict:
        return self._call("GET", f"/sessions/{sid}")

    def decide(self, sid: str, vid: str, decision: str, reason: str | None) -> dict:
        return self._call(
            "PUT", f"/sessions/{sid}/variations/{vid}/decision", json={"decision": decision, "reason": reason}
        )

    def summary(self, sid: str) -> dict:
        return self._call("POST", f"/sessions/{sid}/summary")

    def preview_next_round(self, sid: str, note: str | None) -> dict:
        return self._call("POST", f"/sessions/{sid}/next-round/preview", json={"adjustment_note": note})

    def next_round(self, sid: str, note: str | None) -> dict:
        return self._call("POST", f"/sessions/{sid}/next-round", json={"adjustment_note": note})


def _clean_reason(reason: str | None) -> str | None:
    """Aproxima la normalización de la API (una línea, espacios colapsados) para comparar."""
    return " ".join((reason or "").split()) or None


def pending_decisions(variations: list[dict], decisions, reasons) -> tuple[list[tuple[str, str, str | None]], list[int]]:
    """Compara el formulario con lo persistido.

    Devuelve las decisiones que difieren del registro, como `(variation_id, decisión, motivo)`,
    y las posiciones que aún no tienen decisión en el formulario.
    """
    to_save, missing = [], []
    for v, d, r in zip(variations, decisions, reasons):
        if not d:
            missing.append(v["position"])
            continue
        reason = _clean_reason(r)
        stored = v["decision"]
        if stored is None or stored["decision"] != d or _clean_reason(stored["reason"]) != reason:
            to_save.append((v["id"], d, reason))
    return to_save, missing


def _form_from(variations: list[dict]) -> list:
    """Valores del formulario [decisión, motivo] × 5 tomados del registro."""
    out = []
    for v in variations:
        d = v["decision"]
        out += [gr.update(value=d["decision"] if d else None), gr.update(value=(d["reason"] or "") if d else "")]
    return out


def _empty_cards() -> list:
    out = []
    for i in range(N):
        out += [gr.update(value=None, label=f"Variación {i + 1}"), gr.update(value=None), gr.update(value="")]
    return out


def build_ui(api_base: str) -> gr.Blocks:
    api = Api(api_base)

    with gr.Blocks(title="ProductShot AI") as demo:
        session_state = gr.State(None)

        gr.Markdown("# ProductShot AI\nFoto casera de producto → 5 ambientaciones para catálogo.")
        gr.Markdown(MOCK_NOTICE)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="filepath", label="Foto del producto (JPG/PNG, máx. 10 MB)",
                                    sources=["upload"])
                mode_in = gr.Radio(MODE_CHOICES, value="intermediate", label="Modo de generación")
                start_btn = gr.Button("1. Crear sesión y segmentar", variant="primary")
            with gr.Column(scale=1):
                mask_out = gr.Image(label="Máscara (azul = fondo que se puede cambiar)", interactive=False)
                with gr.Row():
                    reseg_btn = gr.Button("Reintentar segmentación")
                    gen_btn = gr.Button("2. Generar 5 variaciones", variant="primary")
        status_md = gr.Markdown("Sube una foto para empezar.")

        gr.Markdown("## Variaciones\nAcepta o rechaza **cada una**. El motivo es opcional.")
        card_components: list = []  # [img, decisión, motivo] × 5
        decision_inputs: list = []  # [decisión, motivo] × 5
        with gr.Row():
            for i in range(N):
                with gr.Column(min_width=180):
                    img = gr.Image(label=f"Variación {i + 1}", interactive=False, height=260)
                    dec = gr.Radio(DECISION_CHOICES, label="Decisión")
                    reason = gr.Textbox(label="Motivo (opcional)", max_lines=3, max_length=500)
                    card_components += [img, dec, reason]
                    decision_inputs += [dec, reason]

        with gr.Row():
            save_btn = gr.Button("3. Guardar decisiones", variant="primary")
            summary_btn = gr.Button("4. Generar resumen", variant="primary")
        summary_out = gr.Textbox(label="Resumen de decisiones", lines=10, interactive=False)

        gr.Markdown(
            "## Nueva ronda\nCinco variaciones nuevas a partir de lo que aceptaste y de tus motivos, "
            "hasta 3 rondas. La nota se traduce a una lista cerrada de ajustes; nunca se envía tal cual al "
            "generador."
        )
        note_in = gr.Textbox(label="Nota de ajuste (opcional)", max_lines=2, max_length=500,
                             placeholder="Por ejemplo: más luminoso, sobre madera")
        with gr.Row():
            preview_btn = gr.Button("Ver qué se entendió")
            round_btn = gr.Button("5. Crear la siguiente ronda", variant="primary")
        # Texto plano: el plan no se interpreta como Markdown.
        plan_out = gr.Textbox(label="Plan de la siguiente ronda", lines=8, interactive=False)

        # --- manejadores ---

        def start(path, mode):
            if not path:
                return None, None, "Primero sube una foto.", *_empty_cards(), "", ""
            try:
                session = api.create_session(path, mode)
                seg = api.segment(session["id"])
                mask = api.image(seg["mask"]["preview_url"])
            except ApiError as exc:
                return None, None, f"**Error:** {exc}", *_empty_cards(), "", ""
            note = " (máscara simulada)" if seg["mask"]["simulated"] else ""
            return (session["id"], mask, f"Ronda 1 creada en modo `{mode}`. Máscara lista{note}.",
                    *_empty_cards(), "", "")

        def resegment(sid):
            if not sid:
                return None, "No hay sesión activa."
            try:
                seg = api.segment(sid)
                mask = api.image(seg["mask"]["preview_url"])
            except ApiError as exc:
                return gr.update(), f"**Error:** {exc}"
            return mask, "Máscara recalculada."

        def generate(sid):
            if not sid:
                return "No hay sesión activa.", *_empty_cards()
            try:
                api.generate(sid)
                deadline = time.monotonic() + POLL_TIMEOUT_S
                session = api.session(sid)
                while session["status"] == "GENERATING" and time.monotonic() < deadline:
                    time.sleep(0.5)
                    session = api.session(sid)
            except ApiError as exc:
                return f"**Error:** {exc}", *_empty_cards()
            if session["status"] == "GENERATING":
                return "La generación sigue en curso; vuelve a pulsar «Generar» para consultar.", *_empty_cards()
            if session["status"] == "FAILED":
                return f"**La generación falló:** {session.get('failure_reason')}", *_empty_cards()
            cards = []
            try:
                for v in session["variations"]:
                    label = f"{v['position']}. {v.get('label') or STYLE_LABELS.get(v['style'], v['style'])}"
                    if session["simulated"]:
                        label += " · SIMULADA"
                    d = v["decision"]
                    cards += [
                        gr.update(value=api.image(v["image_url"]), label=label),
                        gr.update(value=d["decision"] if d else None),
                        gr.update(value=(d["reason"] or "") if d else ""),
                    ]
            except ApiError as exc:
                return f"**Error:** {exc}", *_empty_cards()
            return (f"Ronda {session['round_index']}: cinco variaciones listas. Revisa cada una.", *cards)

        def save(sid, *values):
            if not sid:
                return "No hay sesión activa."
            try:
                session = api.session(sid)
            except ApiError as exc:
                return f"**Error:** {exc}"
            if not session["variations"]:
                return "Todavía no hay variaciones para decidir."
            to_save, missing = pending_decisions(session["variations"], values[0::2], values[1::2])
            try:
                for vid, d, r in to_save:
                    api.decide(sid, vid, d, r)
            except ApiError as exc:
                return f"**Error:** {exc}"
            if missing:
                return (f"Decisiones al día ({len(to_save)} cambios guardados). "
                        f"Faltan las variaciones {', '.join(map(str, missing))}.")
            return "Se guardaron las cinco decisiones. Ya puedes generar el resumen."

        def summarize(sid, *values):
            # El resumen congela lo PERSISTIDO. Para que coincida con lo que la persona ve,
            # primero se guardan los cambios pendientes del formulario y, al final, el
            # formulario se rellena desde el registro.
            keep = [gr.update()] * (2 * N)
            if not sid:
                return "No hay sesión activa.", "", *keep
            try:
                session = api.session(sid)
                if not session["variations"]:
                    return "Todavía no hay variaciones para resumir.", "", *keep
                to_save, missing = pending_decisions(session["variations"], values[0::2], values[1::2])
                note = ""
                if session["summary_available"]:
                    if to_save or missing:
                        note = (" Los cambios del formulario no se aplicaron: la sesión ya estaba resumida"
                                " y el resumen refleja las decisiones guardadas.")
                else:
                    if missing:
                        return (f"Faltan decisiones en las variaciones {', '.join(map(str, missing))};"
                                " complétalas antes de resumir.", "", *keep)
                    for vid, d, r in to_save:
                        api.decide(sid, vid, d, r)
                s = api.summary(sid)
                frozen = api.session(sid)
            except ApiError as exc:
                return f"**Error:** {exc}", "", *keep
            meta = f"\n\n[generado por: {s['generated_by']} · validación: {s['validation_status']}]"
            if s["limitations"]:
                meta += "\nLimitaciones:\n" + "\n".join(f"- {x}" for x in s["limitations"])
            return ("Resumen generado. Las decisiones de esta sesión quedan congeladas." + note,
                    s["summary"] + meta, *_form_from(frozen["variations"]))

        def preview_round(sid, note):
            if not sid:
                return "No hay sesión activa.", ""
            try:
                plan = api.preview_next_round(sid, (note or "").strip() or None)
            except ApiError as exc:
                return f"**Error:** {exc}", ""
            return "Este es el plan; si te parece bien, crea la siguiente ronda.", "\n".join(plan["explanation"])

        def create_round(sid, note):
            unchanged = [gr.update()] * (3 * N)
            if not sid:
                return sid, gr.update(), "No hay sesión activa.", *unchanged, gr.update(), gr.update(), gr.update()
            try:
                child = api.next_round(sid, (note or "").strip() or None)
                mask = api.image(child["mask"]["preview_url"])
            except ApiError as exc:
                return sid, gr.update(), f"**Error:** {exc}", *unchanged, gr.update(), gr.update(), gr.update()
            status = (f"Ronda {child['round_index']} de 3 creada con la misma foto y máscara. "
                      "Pulsa «2. Generar 5 variaciones».")
            return (child["id"], mask, status, *_empty_cards(), "", "\n".join(child["round_plan"]["explanation"]), "")

        start_btn.click(start, [image_in, mode_in],
                        [session_state, mask_out, status_md, *card_components, summary_out, plan_out])
        preview_btn.click(preview_round, [session_state, note_in], [status_md, plan_out])
        round_btn.click(create_round, [session_state, note_in],
                        [session_state, mask_out, status_md, *card_components, summary_out, plan_out, note_in])
        reseg_btn.click(resegment, [session_state], [mask_out, status_md])
        gen_btn.click(generate, [session_state], [status_md, *card_components])
        save_btn.click(save, [session_state, *decision_inputs], [status_md])
        summary_btn.click(summarize, [session_state, *decision_inputs],
                          [status_md, summary_out, *decision_inputs])

    return demo
