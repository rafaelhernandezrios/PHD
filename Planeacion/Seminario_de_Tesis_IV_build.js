const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/rafael/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/ba8bc45c-a17a-4854-8e8c-85fda9ab82f1/576d0870-4aab-40da-8f70-4f17dd06dadf/skills/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "Seminario_de_Tesis_IV.pptx";
const FECHA = "Octubre 2026";
const SEM = "Seminario de Tesis IV";
const FOOT = `${SEM}  ·  E. Rafael Hernández Ríos  ·  ${FECHA}`;

const THEME = {
  name: "Seminario Tesis",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0E1D33", lt1: "FFFFFF", dk2: "1B2733", lt2: "F1F5F8",
    accent1: "1391B5", accent2: "2E9E6B", accent3: "C97B21",
    accent4: "5D6E7E", accent5: "1B3A5C", accent6: "B5473A",
    hlink: "1391B5", folHlink: "5D6E7E",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = `${SEM} — Sistema Neuroadaptativo para Entrenamiento en Cirugía Robótica`;
pres.author = "E. Rafael Hernández Ríos";
const C = pres.SchemeColor;

const W = 13.333, M = 0.6, CW = W - 2 * M;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.text1 },
  objects: [
    { ellipse: { x: 10.2, y: -1.6, w: 4.8, h: 4.8, fill: { color: C.accent5 }, line: { color: C.accent5 } } },
    { ellipse: { x: 11.4, y: 4.9, w: 3.6, h: 3.6, fill: { color: C.accent5, transparency: 25 }, line: { color: C.accent5 } } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: C.background1 },
  margin: [0.5, M, 0.7, M],
  objects: [
    { text: { text: FOOT, options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: "9AAAB8", fontFace: "Calibri", margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "9AAAB8", align: "right" },
  placeholders: [],
});
pres.defineSlideMaster({
  title: "CIERRE",
  background: { color: C.text1 },
  objects: [
    { ellipse: { x: 11.1, y: 5.3, w: 3.6, h: 3.6, fill: { color: C.accent5, transparency: 30 }, line: { color: C.accent5 } } },
    { text: { text: `M. Eng. Edgar Rafael Hernández Ríos  ·  ${SEM}  ·  ${FECHA}`, options: { x: M, y: 6.95, w: 9, h: 0.3, fontSize: 10, color: "9AAAB8", fontFace: "Calibri", margin: 0 } } },
  ],
});

// ---------- helpers ----------
let n = 0;
function content(section, kicker, title, opts = {}) {
  const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { x: M, y: 0.32, w: CW, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, margin: 0, isTextBox: true, objectName: "Kicker" });
  s.addText(title, { x: M, y: 0.62, w: CW, h: 0.85, fontFace: "Cambria", fontSize: 34, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "Titulo" });
  if (opts.lead) s.addText(opts.lead, { x: M, y: 1.55, w: CW, h: 0.45, fontSize: 15, color: C.accent4, margin: 0, isTextBox: true, objectName: "Entrada" });
  return s;
}
function card(s, x, y, w, h, o = {}) {
  const dark = o.dark;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06,
    fill: o.fill || { color: dark ? C.text1 : C.background2 },
    line: { color: dark ? C.text1 : C.accent4, transparency: dark ? 0 : 78, width: 0.75 },
    objectName: o.name || "Tarjeta",
  });
  const pad = 0.25;
  let cy = y + 0.18;
  if (o.head) {
    s.addText(o.head, { x: x + pad, y: cy, w: w - 2 * pad, h: 0.4, fontSize: o.headSize || 17, bold: true, color: o.headColor || (dark ? C.accent1 : C.text1), margin: 0, valign: "top", isTextBox: true });
    cy += o.headH || 0.48;
  }
  if (o.body) {
    s.addText(o.body, { x: x + pad, y: cy, w: w - 2 * pad, h: y + h - cy - 0.12, fontSize: o.bodySize || 14, color: dark ? "D8E2EA" : C.accent4, margin: 0, valign: "top", paraSpaceAfter: 3, isTextBox: true });
  }
  if (o.foot) s.addText(o.foot, { x: x + pad, y: y + h - 0.42, w: w - 2 * pad, h: 0.3, fontSize: 11, italic: true, color: dark ? "9AAAB8" : C.accent4, margin: 0, isTextBox: true });
}
function callout(s, y, label, text, h = 0.9, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: CW, h, rectRadius: 0.06, fill: { color: color || C.text1 }, line: { color: color || C.text1 }, objectName: "Mensaje" });
  s.addText([
    { text: label + "  ", options: { bold: true, color: color ? C.background1 : C.accent1 } },
    { text, options: { color: C.background1 } },
  ], { x: M + 0.4, y, w: CW - 0.8, h, fontSize: 16, valign: "middle", margin: 0, isTextBox: true });
}
function pill(s, x, y, w, text, tone) {
  const map = { ok: C.accent2, info: C.accent1, warn: C.accent3, bad: C.accent6, mute: C.accent4 };
  const col = map[tone];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.21, fill: { color: col, transparency: 86 }, line: { color: C.text1, width: 1 }, objectName: "Estado" });
  s.addText(text, { x, y, w, h: 0.42, fontSize: 12, bold: true, color: col, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function numDot(s, x, y, label, color, d = 0.6) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: "Numero" });
  s.addText(label, { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, h, big, label, sub, color) {
  card(s, x, y, w, h);
  s.addText(big, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.85, fontFace: "Cambria", fontSize: 40, bold: true, color, margin: 0, valign: "middle", isTextBox: true });
  s.addText(label, { x: x + 0.25, y: y + 1.1, w: w - 0.5, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText(sub, { x: x + 0.25, y: y + 1.5, w: w - 0.5, h: h - 1.6, fontSize: 13, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
}


// Gráficas dibujadas con formas: se conservan etiquetas al importar en Canva/Keynote
function colChart(s, x, y, w, h, title, labels, values, colors, fmt) {
  card(s, x, y, w, h);
  s.addText(title, { x: x + 0.3, y: y + 0.15, w: w - 0.6, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const top = y + 0.95, base = y + h - 0.6, ph = base - top;
  [0, 0.25, 0.5, 0.75, 1].forEach((g) => {
    const gy = base - g * ph;
    s.addShape(pres.shapes.LINE, { x: x + 0.9, y: gy, w: w - 1.2, h: 0, line: { color: C.accent4, transparency: 75, width: 0.75 }, objectName: "Rejilla" });
    s.addText(Math.round(g * 100) + "%", { x: x + 0.2, y: gy - 0.15, w: 0.6, h: 0.3, fontSize: 11, color: C.accent4, align: "right", margin: 0, isTextBox: true });
  });
  const slot = (w - 1.2) / labels.length, bw = slot * 0.55;
  labels.forEach((lab, i) => {
    const bx = x + 0.9 + i * slot + (slot - bw) / 2, bh = values[i] * ph;
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: base - bh, w: bw, h: bh, fill: { color: colors[i] }, line: { color: colors[i] }, objectName: "Barra " + lab });
    s.addText(fmt(values[i]), { x: bx - 0.2, y: base - bh - 0.42, w: bw + 0.4, h: 0.38, fontSize: 16, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true });
    s.addText(lab, { x: bx - 0.3, y: base + 0.08, w: bw + 0.6, h: 0.35, fontSize: 14, color: C.text1, align: "center", margin: 0, isTextBox: true });
  });
}
function barChart(s, x, y, w, h, title, labels, values, vmin, vmax, colors) {
  card(s, x, y, w, h);
  s.addText(title, { x: x + 0.3, y: y + 0.15, w: w - 0.6, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const lx = x + 2.3, rx = x + w - 0.9, top = y + 0.8, bot = y + h - 0.55;
  const ticks = [];
  for (let v = vmin; v <= vmax + 1e-9; v += 0.1) ticks.push(+v.toFixed(2));
  ticks.forEach((v) => {
    const gx = lx + (v - vmin) / (vmax - vmin) * (rx - lx);
    s.addShape(pres.shapes.LINE, { x: gx, y: top, w: 0, h: bot - top, line: { color: C.accent4, transparency: 75, width: 0.75 }, objectName: "Rejilla" });
    s.addText(v.toFixed(2), { x: gx - 0.35, y: bot + 0.05, w: 0.7, h: 0.3, fontSize: 11, color: C.accent4, align: "center", margin: 0, isTextBox: true });
  });
  const slot = (bot - top) / labels.length, bh = slot * 0.6;
  labels.forEach((lab, i) => {
    const by = top + i * slot + (slot - bh) / 2, bw = (values[i] - vmin) / (vmax - vmin) * (rx - lx);
    s.addText(lab, { x: x + 0.2, y: by, w: 2.0, h: bh, fontSize: 14, color: C.text1, align: "right", valign: "middle", margin: 0, isTextBox: true });
    s.addShape(pres.shapes.RECTANGLE, { x: lx, y: by, w: bw, h: bh, fill: { color: colors[i] }, line: { color: colors[i] }, objectName: "Barra " + lab });
    s.addText(values[i].toFixed(2), { x: lx + bw + 0.08, y: by, w: 0.7, h: bh, fontSize: 14, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
}

// ================= 0. Portada =================
pres.addSection({ title: "Portada" });
{
  const s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Portada" });
  s.addText("UNIVERSIDAD AUTÓNOMA DEL ESTADO DE HIDALGO  ·  DOCTORADO EN CIENCIAS COMPUTACIONALES", { x: 0.7, y: 0.55, w: 11.5, h: 0.3, fontSize: 11, color: "AFC6D6", charSpacing: 2, margin: 0, isTextBox: true });
  s.addText(SEM, { x: 0.7, y: 1.2, w: 8, h: 0.45, fontSize: 18, bold: true, color: C.accent1, charSpacing: 1, margin: 0, isTextBox: true });
  s.addText("Sistema Neuroadaptativo para Entrenamiento en Cirugía Robótica", { x: 0.7, y: 1.85, w: 9.6, h: 1.4, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Estimación y regulación de carga cognitiva mediante EEG, IA y realidad virtual háptica", { x: 0.7, y: 3.35, w: 9.6, h: 0.45, fontSize: 18, italic: true, color: "AFC6D6", margin: 0, isTextBox: true });
  s.addText("Estado del proyecto — qué cambió desde el Seminario III", { x: 0.7, y: 4.1, w: 9, h: 0.4, fontSize: 16, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addText("M. Eng. Edgar Rafael Hernández Ríos", { x: 0.7, y: 4.85, w: 9, h: 0.4, fontSize: 18, bold: true, color: C.background1, margin: 0, isTextBox: true });
  s.addText([
    { text: "Director: ", options: { bold: true, breakLine: false } }, { text: "Dr. Omar Arturo Domínguez Ramírez (UAEH)", options: { breakLine: true } },
    { text: "Co-director: ", options: { bold: true } }, { text: "Dr. Christian Peñaloza — Mirai Innovation Research Institute, Osaka" },
  ], { x: 0.7, y: 5.35, w: 9.5, h: 0.65, fontSize: 14, color: "D8E2EA", margin: 0, isTextBox: true });
  s.addText(FECHA, { x: 0.7, y: 6.4, w: 4, h: 0.3, fontSize: 13, color: "9AAAB8", margin: 0, isTextBox: true });
  s.addNotes("Seminario IV. Hilo: desde agosto avanzaron los productos, y medir con más rigor mostró que el clasificador detecta actividad pero no nivel de carga. La propuesta es validar y mejorar el modelo antes de pasar al experimento con usuarios.");
}

// ================= Contexto =================
pres.addSection({ title: "Contexto" });
{
  const s = content("Contexto", "Recordatorio", "Pregunta de investigación y arquitectura");
  s.addText("¿Cómo diseñar una consola de entrenamiento que, mediante EEG e IA, detecte la carga cognitiva y modifique en tiempo real el entorno VR para optimizar el aprendizaje quirúrgico?", { x: M, y: 1.55, w: CW, h: 0.7, fontSize: 16, italic: true, color: C.text1, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 2.45, w: 8.0, h: 4.25, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: C.accent4, transparency: 78 }, objectName: "Marco diagrama" });
  s.addImage({ path: "arquitectura.jpg", x: M + 0.15, y: 2.6, w: 7.7, h: 7.7 * 679 / 1500, altText: "Diagrama de la arquitectura B1–B8 del sistema neuroadaptativo", objectName: "Arquitectura" });
  s.addText("Arquitectura original B1–B8: EEG → procesamiento → IA → VR adaptativo → háptica, en bucle cerrado.", { x: M + 0.15, y: 6.25, w: 7.7, h: 0.35, fontSize: 11, italic: true, color: C.accent4, margin: 0, isTextBox: true });
  card(s, 8.9, 2.45, CW - 8.3, 1.3, { head: "Baja carga", headColor: C.accent1, body: "↑ dificultad · ↑ rigidez háptica · ↓ ayudas" });
  card(s, 8.9, 3.92, CW - 8.3, 1.3, { head: "Zona óptima", headColor: C.accent2, body: "Mantener parámetros · feedback sutil" });
  card(s, 8.9, 5.39, CW - 8.3, 1.3, { head: "Alta carga", headColor: C.accent3, body: "↓ dificultad · ↓ rigidez · ↑ guía visual" });
  s.addNotes("Recordatorio breve para el comité: la pregunta y la arquitectura no cambiaron. Lo que cambia en este seminario es qué tan confiable es el bloque B3.");
}
{
  const s = content("Contexto", "Recordatorio", "Arquitectura del sistema: estado por bloque");
  const rows = [
    ["B1", "Adquisición EEG", "AURA 8 canales · LSL · crudo a 250 Hz con marcas", "Operativo", "ok"],
    ["B2", "Procesamiento y features", "Ventanas 4 s / stride 2 s · 218 features · 10.7 ms", "Operativo", "ok"],
    ["B3", "Estado cognitivo (IA)", "Distingue reposo de tarea; no gradúa la carga", "En revisión", "warn"],
    ["B4", "Entorno VR adaptativo", "Unity 6 · Quest 3 · 5 ejercicios de sutura", "En revisión · 70%", "warn"],
    ["B5", "Feedback multisensorial", "Visual / auditivo / vibrotáctil", "Construido", "ok"],
    ["B6", "Consola háptica", "3D Systems Touch · OpenHaptics · 3.3 N · integrada", "En revisión · 60%", "warn"],
    ["B7", "Usuario", "Experimento con residentes / estudiantes", "Pendiente", "warn"],
    ["B8", "Registro y métricas", "Telemetría 20 Hz + traza del controlador", "Operativo", "ok"],
  ];
  const y0 = 1.6, rh = 0.6, gap = 0.08;
  rows.forEach((r, i) => {
    const y = y0 + i * (rh + gap);
    const hl = r[4] === "warn" && r[0] !== "B7";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: CW, h: rh, rectRadius: 0.05, fill: { color: i % 2 ? "F7FAFC" : C.background2 }, line: { color: C.accent4, transparency: hl ? 0 : 78, width: hl ? 1.5 : 0.75 }, objectName: "Fila " + r[0] });
    s.addText(r[0], { x: M + 0.25, y, w: 0.8, h: rh, fontFace: "Cambria", fontSize: 20, bold: true, color: C.accent1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[1], { x: M + 1.2, y, w: 3.4, h: rh, fontSize: 16, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[2], { x: M + 4.7, y, w: 5.4, h: rh, fontSize: 15, color: C.accent4, valign: "middle", margin: 0, isTextBox: true });
    pill(s, M + CW - 2.35, y + 0.09, 2.1, r[3], r[4]);
  });
  s.addNotes("Cambios respecto al Seminario III: B3 pasa a 'En revisión' porque no gradúa la carga. B4 (entorno VR, ~70%) y B6 (consola háptica, integrada, ~60%) también están en revisión: hay que adaptarlos a la nueva plataforma experimental.");
}
{
  const s = content("Contexto", "Desde el Seminario III", "Qué cambió en estos dos meses", { lead: "Avances de producción, un diagnóstico honesto y un cambio de plan." });
  const items = [
    ["Productos", "ISIE 2026 presentado. Versión final de COMRob enviada; se presenta este mes. Artículo de método enviado a SII 2027, en espera de decisión. Propuesta de capítulo de libro enviada.", C.accent2, "1"],
    ["Diagnóstico", "Medido con más rigor, el clasificador distingue reposo de tarea, pero no gradúa la carga: la carga alta casi nunca se reconoce.", C.accent3, "2"],
    ["Cambio de plan", "Antes del experimento adaptativo: una nueva plataforma experimental para generar un dataset propio, con etiquetas validadas con NASA-TLX, y un nuevo artículo.", C.accent1, "3"],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  items.forEach((it, i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.3, cw, 3.4, {});
    numDot(s, x + 0.25, 2.55, it[3], it[2]);
    s.addText(it[0], { x: x + 0.25, y: 3.35, w: cw - 0.5, h: 0.45, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(it[1], { x: x + 0.25, y: 3.85, w: cw - 0.5, h: 1.75, fontSize: 15, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
  });
  callout(s, 6.0, "Mensaje central:", "que la etiqueta refleje la carga del usuario, no solo la tarea que se le impone.", 0.8);
  s.addNotes("Resumen ejecutivo del seminario. Los tres bloques se desarrollan en el resto de la presentación.");
}

// ================= Producción =================
pres.addSection({ title: "Producción" });
{
  const s = content("Producción", "Producción", "Publicaciones y productos académicos");
  const rows = [
    ["ISIE 2026 — IEEE (Nagoya)", "Validación del índice θ/α (Fz/Pz) en paradigmas controlado y ecológico", "Presentado · en espera de publicación en los proceedings IEEE", "Presentado", "ok"],
    ["COMRob 2026", "EEG Cognitive-Load Control of a Robotic Suturing Trainer", "Versión final enviada · se presenta en octubre 2026", "Por presentar", "info"],
    ["IEEE/SICE SII 2027", "EEG Cognitive-Load Estimation for Adaptive Human-Centered Systems", "Enviado · en espera de decisión", "Enviado", "info"],
    ["Wiley — Adv. Human-Computer Interaction", "A Compact EEG-Based Cognitive Load Index", "En revisión tras el reenvío", "En revisión", "warn"],
    ["Capítulo de libro (propuesta)", "Libro: Hybrid AI Systems for Smart Universities", "Propuesta de capítulo enviada el 21 sep 2026 · en espera", "Enviada", "info"],
  ];
  const lw = 8.4, rh = 0.95, g = 0.08;
  rows.forEach((r, i) => {
    const y = 1.6 + i * (rh + g);
    card(s, M, y, lw, rh);
    s.addText(r[0], { x: M + 0.25, y: y + 0.08, w: 5.9, h: 0.32, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(r[1], { x: M + 0.25, y: y + 0.38, w: 5.9, h: 0.27, fontSize: 12, color: C.accent4, margin: 0, isTextBox: true });
    s.addText(r[2], { x: M + 0.25, y: y + 0.64, w: 5.9, h: 0.25, fontSize: 11, italic: true, color: C.accent1, margin: 0, isTextBox: true });
    pill(s, M + lw - 2.15, y + (rh - 0.42) / 2, 1.9, r[3], r[4]);
  });
  const rx = M + lw + 0.3, rw = CW - lw - 0.3;
  card(s, rx, 1.6, rw, 2.55, { head: "Tesis", body: [
    { text: "Cap. 1–5 redactados con resultados (sep 2026)", options: { bullet: true, breakLine: true } },
    { text: "Cap. 6: diseño del experimento, sin datos", options: { bullet: true, breakLine: true } },
    { text: "Faltan las conclusiones", options: { bullet: true } },
  ] });
  card(s, rx, 4.35, rw, 2.3, { head: "Nuevo artículo JCR", headColor: C.accent1, body: "Dataset propio con tareas de movimiento e interacción en VR, etiquetas validadas con NASA-TLX y un modelo que gradúe la carga." });
  s.addNotes("ISIE: presentado, falta que salgan los proceedings de IEEE. COMRob: versión final enviada, se presenta este mes. SII: enviado, en espera de decisión. Wiley: en revisión tras el reenvío. Capítulo de libro: propuesta 'Neuroadaptive Decision Support for Simulation-Based Skill Training in Smart Universities' enviada el 21 de septiembre al libro 'Hybrid AI Systems for Smart Universities' (eds. J. Ruiz Vanoye y O. Díaz Parra, Universidad Politécnica de Pachuca).");
}

// ================= Diagnóstico =================
pres.addSection({ title: "Diagnóstico del clasificador" });
{
  const s = content("Diagnóstico del clasificador", "Diagnóstico · módulo B3", "El clasificador detecta actividad, no nivel de carga", { lead: "Arithmetic/Stroop, 15 sujetos · calibración contra el reposo propio · evaluación en sujetos no vistos (LOSO)." });
  // Definición de los tres niveles
  s.addText([
    { text: "Niveles:  ", options: { bold: true, color: C.text1 } },
    { text: "Baja", options: { bold: true, color: C.accent1 } }, { text: " = reposo   ·   ", options: { color: C.accent4 } },
    { text: "Óptima", options: { bold: true, color: C.accent2 } }, { text: " = dificultad baja y media   ·   ", options: { color: C.accent4 } },
    { text: "Alta", options: { bold: true, color: C.accent6 } }, { text: " = dificultad alta", options: { color: C.accent4 } },
  ], { x: M, y: 2.0, w: CW, h: 0.4, fontSize: 15, margin: 0, isTextBox: true, objectName: "Definicion niveles" });
  const sw = (CW - 3 * 0.3) / 4, y = 2.6, h = 3.4;
  const items = [
    ["0.82", "Exactitud reposo vs. tarea", C.accent1, "Ventanas bien clasificadas como reposo o como tarea.", "La tarea se reconoce el 93% de las veces; el reposo, solo el 56%."],
    ["0.51", "Exactitud en tres niveles", C.accent3, "Ventanas bien clasificadas como baja, óptima o alta.", "Decir siempre “óptima” daría 0.39: el modelo mejora solo 12 puntos."],
    ["15%", "Recall de carga alta", C.accent6, "De las ventanas de carga alta, cuántas se reconocen como alta.", "De 1,614 ventanas, solo 237. Otras 1,252 se confunden con “óptima”."],
    ["0.40", "Precisión en sobrecarga", C.accent6, "En el lazo cerrado: cuando el sistema detecta sobrecarga, cuántas veces es real.", "6 de cada 10 alarmas son falsas, justo la señal que usa para aliviar."],
  ];
  items.forEach((it, i) => {
    const x = M + i * (sw + 0.3);
    card(s, x, y, sw, h);
    s.addText(it[0], { x: x + 0.25, y: y + 0.15, w: sw - 0.5, h: 0.75, fontFace: "Cambria", fontSize: 38, bold: true, color: it[2], margin: 0, valign: "middle", isTextBox: true });
    s.addText(it[1], { x: x + 0.25, y: y + 0.92, w: sw - 0.5, h: 0.35, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText([
      { text: "Qué mide", options: { bold: true, color: C.text1, breakLine: true } },
      { text: it[3], options: { color: C.accent4 } },
    ], { x: x + 0.25, y: y + 1.35, w: sw - 0.5, h: 0.95, fontSize: 13, margin: 0, valign: "top", isTextBox: true });
    s.addText([
      { text: "Lectura", options: { bold: true, color: C.text1, breakLine: true } },
      { text: it[4], options: { color: C.accent4 } },
    ], { x: x + 0.25, y: y + 2.3, w: sw - 0.5, h: 1.0, fontSize: 13, margin: 0, valign: "top", isTextBox: true });
  });
  callout(s, 6.2, "En resumen:", "el estimador sabe si la persona está haciendo algo, pero no cuánto le cuesta. Para regular la dificultad hace falta lo segundo.", 0.65);
  s.addNotes("Las tres primeras cifras salen del mismo protocolo (Arithmetic/Stroop, 15 sujetos, 218 features, LOSO con calibración). Ojo: 'baja' es el reposo, no una tarea fácil. 0.82: exactitud reposo vs. tarea (csv/protocol_table_218.json). 0.51: exactitud del clasificador jerárquico de tres niveles; la línea base de decir siempre 'óptima' es 0.39 (2,083 de 5,369 ventanas). 15%: recall de carga alta, 237 de 1,614 ventanas; 1,252 se predicen como óptima. 0.40: viene de la prueba del lazo cerrado (paper COMRob): cuando el controlador marca sobrecarga, solo el 40% de las veces la condición era de alta demanda. Recall dice cuántas ventanas de carga alta se detectan; precisión, cuántas alarmas son reales. Ambas salen mal.");
}
{
  const s = content("Diagnóstico del clasificador", "Diagnóstico · módulo B3", "Dónde se pierde la carga alta", { lead: "Proporción de ventanas reconocidas en cada nivel (recall), Random Forest jerárquico, 5,369 ventanas." });
  colChart(s, M, 2.15, 7.2, 4.5, "Recall por nivel de carga", ["Baja", "Óptima", "Alta"], [0.56, 0.76, 0.15], [C.accent1, C.accent2, C.accent6], (v) => Math.round(v * 100) + "%");
  const x = M + 7.5, w = CW - 7.5;
  card(s, x, 2.15, w, 1.4, { head: "Baja: 56%", headColor: C.accent1, body: "La mitad del reposo se lee como tarea." });
  card(s, x, 3.7, w, 1.4, { head: "Óptima: 76%", headColor: C.accent2, body: "Funciona como “cajón” por defecto." });
  card(s, x, 5.25, w, 1.4, { head: "Alta: 15%", headColor: C.accent6, body: "1,252 de 1,614 ventanas de alta carga caen en “óptima”." });
  s.addNotes("Matriz de confusión del clasificador jerárquico de 3 niveles: de 1,614 ventanas de carga alta, 1,252 se predicen como óptima. La frontera óptima–alta no existe de forma nítida con estas features y estos datos.");
}
{
  const s = content("Diagnóstico del clasificador", "Diagnóstico · módulo B3", "Comparación de modelos de clasificación", { lead: "Cinco modelos con el mismo protocolo (reposo vs. tarea, 218 features, LOSO calibrado)." });
  barChart(s, M, 2.15, 7.6, 4.5, "Exactitud reposo vs. tarea por modelo", ["Random Forest", "Gradient Boosting", "MLP", "SVM (RBF)", "Reg. logística"], [0.82, 0.74, 0.70, 0.68, 0.60], 0.5, 0.9, [C.accent1, "AFC6D6", "AFC6D6", "AFC6D6", "AFC6D6"]);
  const x = M + 7.9, w = CW - 7.9;
  card(s, x, 2.15, w, 2.15, { headH: 0.85, head: "Cambiar de modelo no rescata la carga alta", body: "Con las mismas features, ninguno de los cinco supera al Random Forest, y ninguno gradúa mejor." });
  card(s, x, 4.5, w, 2.15, { dark: true, head: "Las causas están antes del modelo", body: "En las etiquetas, en los datos y en las features. Siguiente lámina." });
  s.addNotes("La CNN-LSTM también se probó y no superó al Random Forest. La conclusión de agosto se mantiene: el protocolo pesa más que el modelo. Por eso la mejora tiene que venir de datos, etiquetas y features.");
}
{
  const s = content("Diagnóstico del clasificador", "Diagnóstico · módulo B3", "Cuatro causas probables");
  const rows = [
    ["1", "Etiquetas", "Marcan la dificultad impuesta, no la carga que la persona sintió. Dos personas en el mismo nivel pueden estar en estados distintos.", C.accent1],
    ["2", "Dominio", "Aritmética, Stroop y restas en escritorio. Manipular instrumentos en VR mete movimiento, EMG y otra demanda visual.", C.accent3],
    ["3", "Features", "Solo espectrales y de una ventana suelta: sin contexto temporal, sin covarianza entre canales, sin alinear sujetos.", C.accent2],
    ["4", "Escala", "Ninguno de los dos datasets tiene una escala de carga bien graduada para entrenar “cuánto”, no solo “si”.", C.accent5],
  ];
  const rh = 1.08, g = 0.14;
  rows.forEach((r, i) => {
    const y = 1.6 + i * (rh + g);
    card(s, M, y, CW, rh);
    numDot(s, M + 0.3, y + (rh - 0.6) / 2, r[0], r[3]);
    s.addText(r[1], { x: M + 1.2, y: y, w: 2.4, h: rh, fontSize: 18, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[2], { x: M + 3.6, y: y, w: CW - 3.9, h: rh, fontSize: 15, color: C.accent4, valign: "middle", margin: 0, isTextBox: true });
  });
  callout(s, 6.45, "Implicación:", "cada causa sugiere una línea de trabajo concreta.", 0.42);
  s.addNotes("Las causas 1, 2 y 4 se atacan con la nueva plataforma experimental y el dataset propio. La 3 con nuevas features y modelos.");
}

// ================= Plan =================
pres.addSection({ title: "Plan de trabajo" });
{
  const s = content("Plan de trabajo", "Plan · nueva plataforma experimental", "Una plataforma experimental para un dataset propio", { lead: "Tareas con movimiento e interacción en VR, y etiquetas que reflejen la carga del usuario, no solo la tarea impuesta." });
  const steps = [
    ["1", "Tareas con movimiento", "Manipulación de instrumentos e interacción en VR con niveles de dificultad, en lugar de aritmética o Stroop en escritorio. Registro con el AURA.", C.accent1],
    ["2", "Etiqueta = tarea + NASA-TLX", "La etiqueta combina la condición impuesta con el NASA-TLX de cada bloque. Si la persona no percibe la carga esperada, el dato no se etiqueta como tal.", C.accent3],
    ["3", "Registro sincronizado", "EEG del AURA, telemetría del simulador, marcadores LSL y respuestas NASA-TLX en una sola línea de tiempo.", C.accent2],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  steps.forEach((st, i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.25, cw, 2.75);
    numDot(s, x + 0.25, 2.5, st[0], st[3]);
    s.addText(st[1], { x: x + 1.0, y: 2.5, w: cw - 1.2, h: 0.6, fontSize: 18, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(st[2], { x: x + 0.25, y: 3.3, w: cw - 0.5, h: 1.6, fontSize: 14, color: C.accent4, valign: "top", margin: 0, isTextBox: true });
  });
  const flow = ["Dataset propio", "Entrenamiento del modelo", "Nuevo artículo JCR"];
  const fw = (CW - 2 * 0.6) / 3;
  flow.forEach((f, i) => {
    const x = M + i * (fw + 0.6);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.3, w: fw, h: 0.9, rectRadius: 0.06, fill: { color: C.text1 }, line: { color: C.text1 }, objectName: "Paso " + f });
    s.addText(f, { x, y: 5.3, w: fw, h: 0.9, fontSize: 18, bold: true, color: i === 2 ? C.accent1 : C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    if (i < 2) s.addText("→", { x: x + fw, y: 5.3, w: 0.6, h: 0.9, fontSize: 28, bold: true, color: C.accent1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addNotes("Esta es la respuesta a las causas 1, 2 y 4 del diagnóstico: tareas parecidas al entrenamiento real y etiquetas que reflejan la carga percibida, no solo la impuesta. Del dataset propio salen el modelo y el nuevo artículo.");
}
{
  const s = content("Plan de trabajo", "Plan · plataforma VR", "Plataforma VR lista para medir");
  s.addImage({ path: "banco.jpg", x: M, y: 1.6, w: 4.6, h: 4.6 * 700 / 1086, altText: "Banco de trabajo: laptop, háptico Touch y visor Quest 3", objectName: "Foto banco" });
  s.addImage({ path: "vr.jpg", x: M, y: 4.75, w: 4.6, h: 4.6 * 725 / 1290 * 0.72, sizing: { type: "cover", w: 4.6, h: 1.86 }, altText: "Vista del campo quirúrgico en VR durante el punto simple", objectName: "Foto VR" });
  const x = M + 4.9, w = CW - 4.9;
  s.addText([{ text: "Hoy", options: { bold: true } }], { x: x + 0.25, y: 1.6, w: 3.0, h: 0.35, fontSize: 14, color: C.accent4, margin: 0, isTextBox: true });
  s.addText([{ text: "Lo que se agrega", options: { bold: true } }], { x: x + 3.55, y: 1.6, w: 4.0, h: 0.35, fontSize: 14, color: C.accent1, margin: 0, isTextBox: true });
  const rows = [
    ["Score de carga por UDP, sin marcas de tiempo", "LSL con marcadores sincronizados con el EEG"],
    ["Telemetría exportada al terminar el ejercicio", "Telemetría en vivo como stream LSL"],
    ["Solo los cinco ejercicios de sutura", "Tareas de la nueva plataforma y NASA-TLX dentro del visor"],
    ["Condiciones configuradas a mano", "Gestor de sesión: bloques, orden aleatorizado, registro unificado"],
  ];
  rows.forEach((r, i) => {
    const y = 2.05 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.0, rectRadius: 0.05, fill: { color: C.background2 }, line: { color: C.accent4, transparency: 78 }, objectName: "Fila VR" });
    s.addText(r[0], { x: x + 0.25, y, w: 2.95, h: 1.0, fontSize: 14, color: C.accent4, valign: "middle", margin: 0, isTextBox: true });
    s.addText("→", { x: x + 3.15, y, w: 0.4, h: 1.0, fontSize: 20, bold: true, color: C.accent1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[1], { x: x + 3.55, y, w: w - 3.8, h: 1.0, fontSize: 14, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Unity 6 · OpenXR · Meta Quest 3 · 3D Systems Touch", { x, y: 6.62, w, h: 0.28, fontSize: 12, italic: true, color: C.accent4, margin: 0, isTextBox: true });
  s.addNotes("Estos cambios son los que llevan al entorno VR (B4) y a la consola háptica (B6) de 'en revisión' a operativos para la nueva plataforma. Medir dentro del visor responde además a la crítica de los revisores de COMRob sobre el cambio de dominio escritorio→VR.");
}
{
  const s = content("Plan de trabajo", "Plan · producción", "Un nuevo artículo en revista JCR");
  card(s, M, 1.6, CW, 1.25, { dark: true, head: "Propuesta de contribución", body: "Estimación de carga cognitiva graduada con EEG portátil (AURA), entrenada con un dataset propio de tareas con movimiento e interacción en VR, con etiquetas que combinan la condición impuesta y la carga percibida (NASA-TLX).", bodySize: 15 });
  const cw = (CW - 2 * 0.3) / 3, y = 3.05, h = 2.45;
  card(s, M, y, cw, h, { head: "Qué es nuevo frente a SII", body: [
    { text: "Dataset propio en VR, no tareas de escritorio", options: { bullet: true, breakLine: true } },
    { text: "Etiquetas validadas con NASA-TLX", options: { bullet: true, breakLine: true } },
    { text: "Carga graduada, no solo reposo vs. tarea", options: { bullet: true, breakLine: true } },
    { text: "Medido en el AURA", options: { bullet: true } },
  ] });
  card(s, M + cw + 0.3, y, cw, h, { head: "Revistas candidatas", body: [
    { text: "IEEE Trans. Neural Systems and Rehabilitation Eng. (Q1)", options: { bullet: true, breakLine: true } },
    { text: "Journal of Neural Engineering (Q1)", options: { bullet: true, breakLine: true } },
    { text: "IEEE Trans. Human-Machine Systems", options: { bullet: true, breakLine: true } },
    { text: "Biomedical Signal Processing and Control", options: { bullet: true } },
  ] });
  card(s, M + 2 * (cw + 0.3), y, cw, h, { head: "Cuidado", headColor: C.accent3, body: "Debe citar el artículo de SII y dejar explícito el contenido nuevo: las revistas IEEE piden aportación sustancial sobre la versión de congreso." });
  callout(s, 5.75, "Por qué así:", "con los resultados actuales tal cual, una revista lo leería como una extensión menor de SII. El dataset propio en VR es la aportación nueva.", 0.95);
  s.addNotes("El nuevo artículo es el producto natural de la plataforma experimental: el dataset, el modelo y la publicación salen del mismo trabajo.");
}
{
  const s = content("Plan de trabajo", "Cronograma propuesto", "Próximos nueve meses");
  const cols = [
    ["Oct – Dic 2026", "Plataforma experimental", C.accent1, ["Diseño de tareas con movimiento e interacción VR", "NASA-TLX y marcadores LSL en la plataforma", "Piloto interno de la plataforma", "Seguimiento de SII, Wiley y capítulo de libro"]],
    ["Ene – Mar 2027", "Dataset y modelo", C.accent2, ["Adquisición del dataset propio con el AURA", "Etiquetado combinado: tarea + NASA-TLX", "Entrenamiento del modelo de carga graduada", "Comparación con los datasets públicos"]],
    ["Abr – Jun 2027", "Nuevo artículo", C.accent3, ["Redacción y envío del nuevo artículo JCR", "Decisión de paso al experimento adaptativo", "Protocolo al comité de ética", "Piloto del lazo con la diadema puesta"]],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  cols.forEach((c, i) => {
    const x = M + i * (cw + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: cw, h: 1.1, rectRadius: 0.06, fill: { color: c[2], transparency: 85 }, line: { color: C.accent4, transparency: 78 }, objectName: "Cabecera periodo" });
    s.addText(c[0], { x: x + 0.3, y: 1.7, w: cw - 0.6, h: 0.45, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(c[1].toUpperCase(), { x: x + 0.3, y: 2.17, w: cw - 0.6, h: 0.35, fontSize: 12, bold: true, color: c[2], charSpacing: 3, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.8, w: cw, h: 3.0, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent4, transparency: 78 }, objectName: "Cuerpo periodo" });
    s.addText(c[3].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[3].length - 1 } })), { x: x + 0.25, y: 3.0, w: cw - 0.5, h: 2.7, fontSize: 15, color: C.text1, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
  });
  callout(s, 6.0, "Hito inmediato (próximos 3 meses):", "nueva plataforma experimental lista para adquirir el dataset propio.", 0.75);
  s.addNotes("Fechas propuestas, para discutir con el comité. El experimento adaptativo con usuarios se mueve a la segunda mitad de 2027, que sigue dentro del plan original del Año 3.");
}
{
  const s = content("Plan de trabajo", "Para el comité", "Decisiones que pongo a consideración");
  const items = [
    ["Mover el experimento adaptativo después del dataset propio", "Retrasa el piloto del lazo unos meses, pero evita gastar participantes en un estimador que no gradúa la carga.", C.accent1],
    ["Etiquetar con la carga percibida, no solo con la tarea impuesta", "Combinar la condición con el NASA-TLX de cada bloque, y acordar cómo tratar los casos en que no coinciden.", C.accent2],
    ["Fijar el criterio de paso al experimento", "Un umbral de reconocimiento de carga alta en sujetos no vistos, medido en el AURA y acordado antes de medir.", C.accent3],
  ];
  const rh = 1.35, g = 0.18;
  items.forEach((it, i) => {
    const y = 1.6 + i * (rh + g);
    card(s, M, y, CW, rh);
    numDot(s, M + 0.3, y + (rh - 0.6) / 2, String(i + 1), it[2]);
    s.addText(it[0], { x: M + 1.2, y: y + 0.18, w: CW - 1.5, h: 0.45, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(it[1], { x: M + 1.2, y: y + 0.66, w: CW - 1.5, h: 0.6, fontSize: 15, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Lámina para abrir la discusión con el comité. La segunda y la tercera conviene acordarlas antes de adquirir datos: definen qué cuenta como carga alta y cuándo el sistema está listo.");
}

// ================= Cierre =================
pres.addSection({ title: "Conclusiones" });
{
  const s = pres.addSlide({ masterName: "CIERRE", sectionTitle: "Conclusiones" });
  s.addText("Conclusiones", { x: 0.7, y: 0.75, w: 11, h: 0.8, fontFace: "Cambria", fontSize: 36, bold: true, color: C.background1, margin: 0, isTextBox: true });
  const items = [
    ["01", "El 0.82 no basta para regular la dificultad.", "Distingue reposo de tarea, pero en tres niveles queda en 0.51 y la carga alta se reconoce solo 15% de las veces. El controlador necesita graduar."],
    ["02", "La mejora tiene que venir de antes del modelo.", "Cinco modelos con las mismas features quedan por debajo del Random Forest. Hay que cambiar datos, etiquetas y features."],
    ["03", "Medir la carga del usuario, no solo la de la tarea.", "Una plataforma con tareas de movimiento e interacción en VR y etiquetas validadas con NASA-TLX genera el dataset propio, el modelo y el nuevo artículo."],
  ];
  const rh = 1.35, g = 0.2;
  items.forEach((it, i) => {
    const y = 1.95 + i * (rh + g);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: 11.9, h: rh, rectRadius: 0.06, fill: { color: "15294A" }, line: { color: "24405E", width: 0.75 }, objectName: "Conclusion " + it[0] });
    s.addText(it[0], { x: 1.0, y: y + 0.15, w: 0.9, h: 0.6, fontFace: "Cambria", fontSize: 28, bold: true, color: C.accent1, margin: 0, isTextBox: true });
    s.addText(it[1], { x: 2.1, y: y + 0.18, w: 10.2, h: 0.45, fontSize: 18, bold: true, color: C.background1, margin: 0, isTextBox: true });
    s.addText(it[2], { x: 2.1, y: y + 0.66, w: 10.2, h: 0.6, fontSize: 14.5, color: "B9CEDC", margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Cierre. Lo construido sigue en pie; lo que cambia es el orden: primero un dataset propio con etiquetas que reflejen la carga del usuario y un modelo que la gradúe, después el experimento adaptativo.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("ok", OUT);
})();
