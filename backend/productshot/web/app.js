/* ProductShot: flujo de la web app sobre la API REST (/api/v1). Sin dependencias. */
"use strict";

const API = "/api/v1";
const MAX_SIDE = 2048; // la API trabaja a 1024 px; subir más solo gasta datos
const EXPECTED_S = 75; // duración típica de un lote (prueba de humo: 45–120 s)
const POLL_MS = 2500;
const MAX_WAIT_S = 600; // G14

const $ = (id) => document.getElementById(id);
const state = { blob: null, previewUrl: null, category: null, mode: "intermediate", session: null, rounds: 0 };

// ---------- utilidades ----------

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

let toastTimer = null;
function toast(msg, ms = 6000) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

async function api(path, { method = "GET", body, json } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (body) {
    opts.body = body;
  }
  let res;
  try {
    res = await fetch(API + path, opts);
  } catch {
    throw new Error("No hay conexión con el servidor. Revisa tu internet e inténtalo de nuevo.");
  }
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch { /* sin cuerpo */ }
  if (!res.ok) {
    const msg = data?.error?.message || data?.detail || `Algo salió mal (error ${res.status}).`;
    const err = new Error(msg);
    err.status = res.status;
    err.code = data?.error?.code;
    throw err;
  }
  return data;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function show(screen) {
  for (const s of document.querySelectorAll(".screen")) s.hidden = s.id !== `s-${screen}`;
  const order = ["foto", "crear", "elegir", "listo"];
  const i = order.indexOf(screen);
  document.querySelectorAll(".steps li").forEach((li, j) => {
    li.classList.toggle("done", j < i);
    li.classList.toggle("now", j === i);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- 1. foto ----------

async function toJpeg(file) {
  // Reduce y convierte a JPEG en el celular (también las HEIC de iPhone). La orientación
  // EXIF la aplica el navegador al dibujar. Si no se puede decodificar, se envía el original.
  try {
    const url = URL.createObjectURL(file);
    const img = await new Promise((ok, fail) => {
      const i = new Image();
      i.onload = () => ok(i);
      i.onerror = fail;
      i.src = url;
    });
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(img.naturalWidth * scale);
    c.height = Math.round(img.naturalHeight * scale);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    const blob = await new Promise((ok) => c.toBlob(ok, "image/jpeg", 0.9));
    return blob || file;
  } catch {
    return file;
  }
}

async function onPick(ev) {
  const file = ev.target.files?.[0];
  ev.target.value = "";
  if (!file) return;
  state.blob = await toJpeg(file);
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(state.blob);
  $("preview-img").src = state.previewUrl;
  $("picker").hidden = true;
  $("preview").hidden = false;
  $("category-group").hidden = false;
  $("mode-group").hidden = false;
  $("foto-actions").hidden = false;
  updateCreate();
}

function updateCreate() {
  $("btn-create").disabled = !(state.blob && state.category);
  $("btn-create").textContent = state.category ? "Crear mis fotos" : "Elige qué vendes";
}

function pickChip(btn, group, key) {
  for (const b of group.querySelectorAll("[role=radio]")) b.setAttribute("aria-checked", String(b === btn));
  state[key] = btn.dataset[key === "category" ? "cat" : "mode"];
  updateCreate();
}

// ---------- 2. crear ----------

let waitTimer = null;
function startWaiting(firstMsg) {
  show("crear");
  $("wait-img").src = state.previewUrl || "";
  $("wait-msg").textContent = firstMsg;
  $("wait-bar").style.width = "4%";
  $("wait-time").textContent = "Suele tardar alrededor de un minuto.";
  const t0 = Date.now();
  const msgs = [
    [0, "Creando cinco escenas para tu producto…"],
    [20, "Ajustando la luz de cada escena…"],
    [40, "Revisando que tu producto se vea igual que en tu foto…"],
    [70, "Casi listo…"],
    [120, "Está tardando un poco más de lo normal: seguimos revisando cada foto."],
  ];
  clearInterval(waitTimer);
  waitTimer = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    if (!state.generating) return; // revisando la foto o encendiendo el servidor: sin avance falso
    const g = (Date.now() - state.genStart) / 1000;
    const pct = Math.min(92, 4 + (g / EXPECTED_S) * 88);
    $("wait-bar").style.width = `${pct}%`;
    const m = msgs.filter(([at]) => g >= at).pop();
    $("wait-msg").textContent = m[1];
    if (s > 30) $("wait-time").textContent = `Llevamos ${Math.round(s)} s.`;
  }, 1000);
}
function stopWaiting() { clearInterval(waitTimer); state.generating = false; $("wait-bar").style.width = "100%"; }

async function pollReady(sessionId) {
  const t0 = Date.now();
  while ((Date.now() - t0) / 1000 < MAX_WAIT_S) {
    const s = await api(`/sessions/${sessionId}`);
    if (s.status === "READY_FOR_REVIEW" || s.status === "REVIEWED") return s;
    if (s.status === "FAILED") throw new Error(s.failure_reason || "No pudimos crear las fotos. Inténtalo con otra foto.");
    await sleep(POLL_MS);
  }
  throw new Error("Está tardando demasiado. Inténtalo de nuevo en unos minutos.");
}

async function generateAndShow(session) {
  state.generating = true;
  state.genStart = Date.now();
  await api(`/sessions/${session.id}/generate`, { method: "POST" });
  const ready = await pollReady(session.id);
  stopWaiting();
  state.session = ready;
  renderCards(ready);
  show("elegir");
}

async function create() {
  startWaiting("Revisando tu foto…");
  // Mientras se revisa la foto (G2, G3 y G15) no se muestra el avance de la generación: si el
  // servidor está apagado, encenderlo toma hasta un par de minutos y conviene decirlo.
  const coldTimer = setTimeout(() => {
    if (!state.generating) {
      $("wait-msg").textContent = "Encendiendo el servidor: la primera foto tarda un poco más (hasta 2 minutos).";
    }
  }, 15000);
  try {
    const fd = new FormData();
    fd.append("image", state.blob, "producto.jpg");
    fd.append("mode", state.mode);
    fd.append("category", state.category);
    const session = await api("/sessions", { method: "POST", body: fd });
    $("wait-msg").textContent = "Separando tu producto del fondo…";
    await api(`/sessions/${session.id}/segment`, { method: "POST" });
    const fresh = await api(`/sessions/${session.id}`);
    state.notes = (fresh.warnings || []).concat(fresh.mask?.warnings || []);
    if (fresh.simulated) toast("Modo de prueba: el servidor usa imágenes simuladas.");
    state.rounds = 1;
    clearTimeout(coldTimer);
    await generateAndShow(fresh);
  } catch (e) {
    clearTimeout(coldTimer);
    stopWaiting();
    show("foto");
    toast(e.message, 9000);
  }
}

// ---------- 3. elegir ----------

function renderCards(session) {
  state.locked = false;
  const box = $("cards");
  box.innerHTML = "";
  const round = session.round_index || 1;
  $("round-badge").hidden = round === 1;
  $("round-badge").textContent = `Ronda ${round} de 3`;
  const notes = state.notes?.length ? state.notes : [];
  if (notes.length && round === 1) {
    const n = document.createElement("p");
    n.className = "note";
    n.textContent = notes.map((w) => w.message).join(" ");
    box.appendChild(n);
  }
  const vars = [...session.variations].sort((a, b) => a.position - b.position);
  for (const v of vars) {
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.id = v.id;
    card.innerHTML = `
      <img src="${esc(v.image_url)}" alt="Foto ${v.position}: ${esc(v.label)}" loading="lazy">
      <div class="body">
        <div class="name"><span>${v.position}. ${esc(v.label)}</span>
          <a class="save" href="${esc(v.image_url)}" download="productshot-${v.position}.png">Guardar</a></div>
        <div class="choice">
          <button type="button" class="yes" aria-pressed="false">Me gusta</button>
          <button type="button" class="no" aria-pressed="false">No me gusta</button>
        </div>
        <input class="field reason" maxlength="200" hidden placeholder="¿Por qué? (opcional)">
      </div>`;
    const yes = card.querySelector(".yes");
    const no = card.querySelector(".no");
    const reason = card.querySelector(".reason");
    const d = v.decision;
    if (d) {
      (d.decision === "accepted" ? yes : no).setAttribute("aria-pressed", "true");
      reason.hidden = false;
      reason.value = d.reason || "";
    }
    const decide = async (value) => {
      yes.setAttribute("aria-pressed", String(value === "accepted"));
      no.setAttribute("aria-pressed", String(value === "rejected"));
      reason.hidden = false;
      reason.placeholder = value === "accepted" ? "¿Qué te gustó? (opcional)" : "¿Qué no te gustó? (opcional)";
      card.dataset.decision = value;
      await saveDecision(card);
    };
    yes.addEventListener("click", () => decide("accepted"));
    no.addEventListener("click", () => decide("rejected"));
    reason.addEventListener("change", () => card.dataset.decision && saveDecision(card));
    if (d) card.dataset.decision = d.decision;
    box.appendChild(card);
  }
  updateCounter();
}

function saveDecision(card) {
  // Tras el resumen las decisiones quedan congeladas (la API respondería 409): no se reenvían.
  if (state.locked) return Promise.resolve();
  const reason = card.querySelector(".reason").value.trim();
  const decision = card.dataset.decision;
  // Los guardados de una misma tarjeta van en orden; el resumen espera a que terminen todos.
  card._pending = (card._pending || Promise.resolve()).then(async () => {
    try {
      await api(`/sessions/${state.session.id}/variations/${card.dataset.id}/decision`, {
        method: "PUT", json: { decision, reason: reason || null },
      });
      card.dataset.saved = "1";
      card.dataset.savedReason = reason;
    } catch (e) {
      card.dataset.saved = "";
      if (!state.locked) toast(e.message);
    }
    updateCounter();
  });
  return card._pending;
}

function updateCounter() {
  const cards = [...document.querySelectorAll("#cards .card")];
  const done = cards.filter((c) => c.dataset.saved === "1" || (c.dataset.decision && c.dataset.saved !== "")).length;
  $("counter").textContent = `${done} de ${cards.length} elegidas`;
  $("btn-summary").disabled = done < cards.length;
}

// ---------- 4. resumen ----------

async function summary() {
  const btn = $("btn-summary");
  btn.disabled = true;
  btn.textContent = "Preparando…";
  try {
    // Guarda los motivos escritos que aún no se enviaron (la persona puede tocar «Ver resumen»
    // sin salir del campo) y espera a que terminen todos los guardados antes de congelar.
    const cards = [...document.querySelectorAll("#cards .card")];
    for (const c of cards) {
      if (c.querySelector(".reason").value.trim() !== (c.dataset.savedReason || "")) saveDecision(c);
    }
    await Promise.all(cards.map((c) => c._pending || Promise.resolve()));
    state.locked = true;
    const s = await api(`/sessions/${state.session.id}/summary`, { method: "POST" });
    const session = await api(`/sessions/${state.session.id}`);
    state.session = session;
    renderSummary(s, session);
    show("listo");
  } catch (e) {
    state.locked = false;
    toast(e.message);
  } finally {
    btn.textContent = "Ver resumen";
    updateCounter();
  }
}

function renderSummary(s, session) {
  const paras = String(s.summary || "").split(/\n{2,}/).map((p) => `<p>${esc(p).replace(/\n/g, "<br>")}</p>`).join("");
  const who = {
    llm_validated: "Escrito por el asistente de IA y verificado contra tus decisiones.",
    hybrid: "Escrito por el asistente de IA (en parte) y verificado contra tus decisiones.",
    deterministic_fallback: "Resumen automático, armado directamente a partir de tus decisiones.",
  }[s.generated_by] || "";
  $("summary").innerHTML = paras + (who ? `<p class="tag">${esc(who)}</p>` : "");
  const chosen = $("chosen");
  const accepted = session.variations.filter((v) => s.accepted_ids.includes(v.id)).sort((a, b) => a.position - b.position);
  chosen.innerHTML = accepted.length
    ? accepted.map((v) => `<a href="${esc(v.image_url)}" download="productshot-${v.position}.png" title="Guardar"><img src="${esc(v.image_url)}" alt="${esc(v.label)}"></a>`).join("")
    : `<p class="empty">No elegiste ninguna. Prueba otra ronda: usamos lo que no te gustó para mejorar.</p>`;
  $("next-group").hidden = (session.round_index || 1) >= 3;
  $("next-note").value = "";
}

async function nextRound() {
  const note = $("next-note").value.trim();
  startWaiting("Preparando una ronda con tus gustos…");
  try {
    const child = await api(`/sessions/${state.session.id}/next-round`, { method: "POST", json: { adjustment_note: note || null } });
    state.notes = [];
    await generateAndShow(child);
  } catch (e) {
    stopWaiting();
    show("listo");
    toast(e.message, 9000);
  }
}

async function removeAll() {
  if (!state.session) return reset();
  if (!confirm("¿Borrar tu foto y las fotos creadas del servidor? No se puede deshacer.")) return;
  try {
    await api(`/sessions/${state.session.id}`, { method: "DELETE" });
    reset();
    toast("Listo: borramos tus fotos del servidor.");
  } catch (e) {
    toast(e.message);
  }
}

function reset() {
  state.blob = null;
  state.session = null;
  state.notes = [];
  state.category = null;
  $("preview").hidden = true;
  $("picker").hidden = false;
  $("category-group").hidden = true;
  $("mode-group").hidden = true;
  $("foto-actions").hidden = true;
  document.querySelectorAll("#category-group [role=radio]").forEach((b) => b.setAttribute("aria-checked", "false"));
  updateCreate();
  show("foto");
}

// ---------- enlaces ----------

$("in-camera").addEventListener("change", onPick);
$("in-gallery").addEventListener("change", onPick);
// Se abren desde el botón (gesto directo de la persona), como exigen iOS Safari y Chrome móvil.
$("btn-camera").addEventListener("click", () => $("in-camera").click());
$("btn-gallery").addEventListener("click", () => $("in-gallery").click());
$("btn-change").addEventListener("click", () => { $("picker").hidden = false; $("preview").hidden = true; });
for (const b of document.querySelectorAll("#category-group [role=radio]")) {
  b.addEventListener("click", () => pickChip(b, $("category-group"), "category"));
}
for (const b of document.querySelectorAll("#mode-group [role=radio]")) {
  b.addEventListener("click", () => pickChip(b, $("mode-group"), "mode"));
}
$("btn-create").addEventListener("click", create);
$("btn-summary").addEventListener("click", summary);
$("btn-next").addEventListener("click", nextRound);
$("btn-new").addEventListener("click", reset);
$("btn-delete").addEventListener("click", removeAll);
show("foto");
updateCreate();

// Plazo del borrado automático (G11), tomado de la configuración del servidor.
fetch(`${API}/healthz`)
  .then((r) => (r.ok ? r.json() : null))
  .then((h) => {
    const hours = h && h.retention_hours;
    if (hours > 0) $("retention").textContent = ` y se borran solas a las ${hours} h`;
  })
  .catch(() => {});
