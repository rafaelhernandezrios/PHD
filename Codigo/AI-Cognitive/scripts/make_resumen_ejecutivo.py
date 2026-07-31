"""Resumen ejecutivo del paper COMRob 2026, en PDF, para el autor y su asesor."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

OUT = (Path(__file__).resolve().parent.parent / "paper" /
       "Resumen_ejecutivo_COMRob2026.pdf")

INK   = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6570")
ACC   = colors.HexColor("#1f4e79")
RULE  = colors.HexColor("#d4d9de")
BAND  = colors.HexColor("#eef2f6")

ss = getSampleStyleSheet()
S = {
    "title":  ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                             fontSize=19, leading=23, textColor=ACC, spaceAfter=2),
    "sub":    ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=16),
    "h1":     ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                             fontSize=13, leading=16, textColor=ACC,
                             spaceBefore=15, spaceAfter=6),
    "h2":     ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=10.5, leading=13.5, textColor=INK,
                             spaceBefore=10, spaceAfter=4),
    "body":   ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=9.6, leading=14, textColor=INK,
                             alignment=TA_JUSTIFY, spaceAfter=7),
    "bullet": ParagraphStyle("bu", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=9.6, leading=13.5, textColor=INK,
                             leftIndent=13, bulletIndent=3, spaceAfter=2.5),
    "note":   ParagraphStyle("n", parent=ss["Normal"], fontName="Helvetica-Oblique",
                             fontSize=9, leading=12.5, textColor=MUTED,
                             leftIndent=9, spaceAfter=7),
    "cell":   ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=8.6, leading=11.4, textColor=INK),
    "cellb":  ParagraphStyle("cb", parent=ss["Normal"], fontName="Helvetica-Bold",
                             fontSize=8.6, leading=11.4, textColor=INK),
    "cellh":  ParagraphStyle("ch", parent=ss["Normal"], fontName="Helvetica-Bold",
                             fontSize=8.6, leading=11.4, textColor=colors.white),
}

E = []
def h1(t):   E.append(Paragraph(t, S["h1"]))
def h2(t):   E.append(Paragraph(t, S["h2"]))
def p(t):    E.append(Paragraph(t, S["body"]))
def note(t): E.append(Paragraph(t, S["note"]))
def gap(h=5): E.append(Spacer(1, h))
def bullets(items):
    for it in items:
        E.append(Paragraph(it, S["bullet"], bulletText="•"))
    gap(6)

def table(rows, widths, header=True, highlight=None):
    """Tabla que no se parte entre paginas."""
    data = []
    for r_i, row in enumerate(rows):
        st = S["cellh"] if (header and r_i == 0) else (
             S["cellb"] if (highlight and r_i in highlight) else S["cell"])
        data.append([Paragraph(str(c), st) for c in row])
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE)]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), ACC),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACC)]
    for r in (highlight or []):
        cmds.append(("BACKGROUND", (0, r), (-1, r), BAND))
    t.setStyle(TableStyle(cmds))
    # arrastra el parrafo/encabezado inmediatamente anterior para que no
    # quede huerfano al pie de una pagina
    prev = E.pop() if E and isinstance(E[-1], Paragraph) else None
    E.append(KeepTogether([prev, t] if prev is not None else [t]))
    gap(9)


# =============================================================== portada
E.append(Paragraph("Control de dificultad por carga cognitiva EEG "
                   "en un entrenador de sutura robótica", S["title"]))
E.append(Paragraph("Resumen ejecutivo del artículo enviado a COMRob 2026 &nbsp;·&nbsp; "
                   "documento interno de trabajo", S["sub"]))

h1("En una frase")
p("El artículo <b>no propone otro clasificador de carga mental</b>. Propone lo que "
  "casi siempre queda sin especificar: una vez estimada la carga cognitiva, "
  "<b>qué debe cambiar el simulador, cuánto, cuándo, y cómo evita reaccionar de "
  "forma inestable al ruido del EEG</b>.")

h1("El problema")
p("Los simuladores quirúrgicos fijan la dificultad antes de empezar el ejercicio, "
  "pero la carga mental del usuario cambia durante la tarea: puede estar poco "
  "exigido y aburrirse, en una zona productiva, o demasiado exigido y empezar a "
  "fallar. La idea es estimar esa carga con EEG y cerrar el ciclo:")
note("usuario &rarr; EEG &rarr; estimación de carga &rarr; controlador &rarr; "
     "cambio en el simulador &rarr; usuario")
p("El obstáculo es que <b>el EEG es ruidoso</b>. Si el simulador reaccionara a cada "
  "predicción, la dificultad cambiaría constantemente y el sistema sería "
  "inutilizable. Por eso el artículo se centra en diseñar la capa de control que "
  "convierte una señal imperfecta en acciones estables.")

h1("Qué sistema se construyó")
p("Un simulador estilo da Vinci en Unity, sobre Meta Quest 3. El usuario controla "
  "dos instrumentos que atraviesan puntos de trócar, replicando la restricción "
  "geométrica de la cirugía laparoscópica. El instrumento derecho puede además "
  "controlarse con un <b>3D Systems Touch</b>, que entrega fuerza real y no solo "
  "vibración.")
h2("Cinco ejercicios")
bullets(["Manipulación de aguja",
         "Precisión mediante anillos",
         "Sutura interrumpida",
         "Sutura continua",
         "Nudo intracorpóreo"])
h2("Telemetría a 20 Hz")
p("Posición de los instrumentos, apertura de las pinzas, longitud de trayectoria, "
  "tiempo de ejecución, errores de punción, tensión excesiva del hilo, caída de la "
  "aguja y precisión en milímetros. <b>Esta telemetría permite saber si el usuario "
  "se está desempeñando bien, con independencia de lo que diga el EEG</b>, y es la "
  "base del <i>performance gate</i>.")

# =============================================================== controlador
h1("Cómo funciona el controlador")
p("Recibe una estimación de carga cada dos segundos, calculada sobre ventanas EEG "
  "de cuatro segundos, en una escala de <b>0 (reposo) a 3 (máxima demanda)</b>. "
  "Mantiene una dificultad global <b>d</b> entre 0 y 1, donde d = 0.5 reproduce "
  "exactamente la configuración estándar del simulador.")
table([["Zona detectada", "Significado", "Acción sobre d"],
       ["under",   "Carga similar a reposo",     "sube 0.08 (gradual)"],
       ["optimal", "Zona de trabajo deseada",    "sin cambio"],
       ["over",    "Condición de alta demanda",  "baja 0.15 (más rápido)"]],
      [3.4*cm, 6.6*cm, 5.2*cm])
note("La asimetría es deliberada: exigir más puede hacerse con cautela, mientras "
     "que aliviar una condición potencialmente excesiva debe ser más rápido.")

h1("Cómo evita oscilar")
p("Esta es una de las contribuciones principales. Cuatro mecanismos se interponen "
  "entre la lectura del EEG y cualquier cambio real:")
table([["Mecanismo", "Qué hace"],
       ["Suavizado exponencial",
        "Promedia con las lecturas anteriores (&lambda; = 0.6); una ventana rara no manda"],
       ["Zona muerta e histéresis",
        "Dentro de la zona óptima no cambia nada, y salir de una zona exige rebasar "
        "el umbral con margen, no rozarlo"],
       ["Persistencia",
        "Una zona nueva debe sostenerse tres ventanas (6 s) antes de aceptarse"],
       ["Periodo refractario",
        "Tras actuar, 10 s de silencio para que el usuario responda al cambio"]],
      [4.6*cm, 10.6*cm])

h1("Qué parámetros adapta, y cuándo")
p("El artículo separa los parámetros en dos grupos, y esa separación es una de las "
  "decisiones de diseño más defendibles del trabajo.")
table([["Continuos &mdash; pueden cambiar <i>durante</i> un ejercicio",
        "De frontera &mdash; solo <i>entre</i> ejercicios"],
       ["Escalamiento de movimiento<br/>Intensidad de la guía háptica<br/>"
        "Tolerancia de tensión del hilo",
        "Tamaño del objetivo de punción<br/>Tamaño de los anillos<br/>"
        "Tiempo objetivo<br/>Longitud de trayectoria permitida<br/>"
        "Presupuesto de errores"]],
      [7.6*cm, 7.6*cm])
p("<b>La razón es metodológica.</b> Cambiar el criterio de evaluación a mitad de un "
  "intento volvería incomparables los resultados. No tendría sentido reducir el "
  "tamaño del objetivo a media sutura y luego comparar ese intento con otro en el "
  "que el objetivo permaneció fijo.")

h1("El performance gate")
p("El EEG por sí solo no distingue dos casos que exigen respuestas opuestas: el "
  "usuario con alta carga <i>que está fallando</i>, y el usuario con alta carga "
  "<i>que se está desempeñando bien</i>. El segundo puede ser esfuerzo productivo, "
  "y bajarle la dificultad sería contraproducente. Por eso el controlador también "
  "consulta la telemetría del simulador:")
table([["Carga estimada", "Desempeño", "Acción"],
       ["Alta demanda", "sin errores recientes", "mantener"],
       ["Alta demanda", "con errores",           "reducir dificultad"],
       ["Tipo reposo",  "buen desempeño",        "aumentar dificultad"],
       ["Tipo reposo",  "con errores",           "no aumentar"],
       ["cualquiera",   "sin telemetría",        "acción conservadora"]],
      [4.4*cm, 5.6*cm, 5.2*cm])
note("En la evaluación offline no existía una secuencia real de desempeño, así que "
     "esta lógica se verificó con casos deterministas. Falta validarla con usuarios.")

# =============================================================== evaluacion
h1("De dónde viene la estimación EEG")
p("Diadema inalámbrica de ocho canales (Fp1, Fp2, F3, <b>Fz</b>, F4, P3, <b>Pz</b>, "
  "P4). Se extraen potencias de banda, razones espectrales, parámetros de Hjorth, "
  "entropía espectral y la relación theta/alpha entre Fz y Pz, todo normalizado "
  "contra un registro de reposo del mismo sujeto.")
p("El controlador usa la salida continua de un <b>regresor ordinal</b> "
  "(gradient boosting) en escala 0&ndash;3. Cada una de las 5,369 estimaciones fue "
  "producida por un modelo que <b>nunca había visto a esa persona</b> durante el "
  "entrenamiento.")

h1("Cómo se evaluó")
p("<b>No</b> hubo todavía un experimento con una persona usando simultáneamente el "
  "simulador, el EEG y el controlador. Se hizo una evaluación "
  "<b>software-in-the-loop</b>: para cada participante se construyó una rampa "
  "reposo &rarr; baja &rarr; media &rarr; alta &rarr; reposo. Los scores EEG dentro "
  "de cada bloque son reales; el orden de los bloques fue impuesto para crear una "
  "rampa controlada de demanda.")
p("Eso permite estudiar oscilaciones, tiempo de reacción, errores de dirección, "
  "estabilidad y coste computacional. <b>No permite afirmar que el sistema mejore "
  "el aprendizaje</b>, y el artículo lo dice explícitamente.")

h1("Resultados principales")

h2("1. Los mecanismos anti-oscilación funcionan")
table([["Guardas activas", "Cambios de zona/min", "Acciones de dificultad/min", "Acuerdo"],
       ["ninguna (score crudo)", "12.16 &plusmn; 1.91", "7.21 &plusmn; 1.20", "0.534 &plusmn; 0.06"],
       ["+ suavizado",           "5.74 &plusmn; 1.09",  "2.91 &plusmn; 0.55", "0.560 &plusmn; 0.07"],
       ["+ zona muerta",         "2.95 &plusmn; 0.99",  "1.50 &plusmn; 0.50", "0.567 &plusmn; 0.08"],
       ["+ persistencia",        "1.15 &plusmn; 0.32",  "0.61 &plusmn; 0.19", "0.572 &plusmn; 0.08"]],
      [4.4*cm, 3.9*cm, 4.2*cm, 2.7*cm], highlight=[4])
p("Se pasó de una acción de dificultad <b>cada ocho segundos</b> a <b>una cada cien</b>, "
  "y el acuerdo con la condición impuesta <i>no bajó</i>. Se eliminaron nueve de "
  "cada diez oscilaciones sin perder información útil.")

h2("2. El sistema es asimétrico")
table([["", "Estados tipo reposo", "Alta demanda"],
       ["Precisión",           "0.89 &plusmn; 0.16", "0.46 &plusmn; 0.25"],
       ["Recall",              "0.67 &plusmn; 0.12", "0.37 &plusmn; 0.33"],
       ["Sujetos detectados",  "15 de 15",           "12 de 15"],
       ["Latencia",            "22.5 s",             "37.7 s"]],
      [4.4*cm, 5.4*cm, 5.4*cm])
p("El EEG distingue bien <b>reposo frente a actividad</b>, pero no distingue de "
  "forma confiable entre demanda baja, media y alta.")

h2("3. El problema está en la percepción, no en la política")
p("Se repitió la prueba con la <b>misma política de control</b> &mdash; mismo "
  "suavizado, misma zona muerta, misma persistencia, mismos pasos &mdash; pero "
  "alimentándola con la <b>condición verdadera impuesta</b> en lugar del score EEG. "
  "Es decir, un sensor perfecto.")
table([["", "Con EEG", "Con entrada perfecta"],
       ["Recall de alta demanda", "0.37", "1.00"],
       ["Sujetos cubiertos",      "12 de 15", "15 de 15"],
       ["Cambios de zona/min",    "1.15", "0.20"],
       ["Acuerdo",                "0.572", "0.683"]],
      [5.4*cm, 4.9*cm, 4.9*cm], highlight=[1])
p("<b>La política responde correctamente cuando la entrada es adecuada.</b> Si el "
  "controlador estuviera mal diseñado, seguiría fallando con un sensor perfecto, y "
  "no falla. La limitación es la capacidad del estimador EEG para separar demanda "
  "media de alta.")
note("Dato secundario revelador: incluso con entrada perfecta el acuerdo solo llega "
     "a 0.683. El techo lo imponen las propias guardas, que retrasan cada transición "
     "por diseño. Eso confirma que el acuerdo debe leerse como medida de "
     "seguimiento, no como precisión de reconocimiento.")

h2("4. Mover el umbral no resuelve nada")
p("Se barrió el umbral superior entre 1.5 y 2.0. En todo ese rango el recall nunca "
  "superó 0.49 y la precisión nunca superó 0.54. <b>El problema no se arregla "
  "ajustando el threshold</b>: la información necesaria no está suficientemente "
  "separada en el score del estimador.")

h2("5. El coste computacional es despreciable")
p("Una actualización completa cuesta <b>16.03 ms</b> (percentil 95: 18.59 ms), menos "
  "del 1% del intervalo de dos segundos entre actualizaciones. El cuello de botella "
  "no es el cómputo sino la ventana EEG de cuatro segundos, la persistencia, el "
  "refractario y la confiabilidad del estimador. <b>La latencia percibida del "
  "sistema está en el orden de 22 a 38 segundos, no en 16 ms.</b>")

# =============================================================== conclusion
h1("La conclusión científica")
p("No es que «el EEG funciona» ni que «el clasificador falla». Es una conclusión "
  "<b>arquitectónica</b>:")
E.append(Table([[Paragraph(
    "No conviene usar una única estimación EEG para controlar ambas direcciones "
    "de adaptación.", S["cellb"])]], colWidths=[15.2*cm],
    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND),
                      ("LEFTPADDING", (0, 0), (-1, -1), 11),
                      ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                      ("TOPPADDING", (0, 0), (-1, -1), 9),
                      ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                      ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACC)])))
gap(10)
bullets(["Usar <b>EEG</b> para detectar estados tipo reposo y decidir cuándo "
         "aumentar gradualmente la dificultad.",
         "Usar principalmente la <b>telemetría de desempeño</b> para decidir cuándo "
         "reducirla.",
         "Usar el EEG solo como evidencia complementaria en la dirección de alivio."])
p("La arquitectura reconoce que los sensores tienen fortalezas distintas y evita "
  "actuar con confianza sobre la salida más débil del estimador. "
  "<b>Un resultado negativo convertido en una recomendación de diseño concreta</b> "
  "suele valer más que ocultar el bajo desempeño de una rama.")

h1("Qué decirle al asesor en dos minutos")
p("El artículo propone un controlador de dificultad adaptativa para un simulador de "
  "sutura robótica en VR. La novedad no está en clasificar carga cognitiva, sino en "
  "convertir una estimación EEG ruidosa en cambios concretos y estables sobre el "
  "simulador: escalamiento de movimiento, guía háptica, tolerancias, tamaños de "
  "objetivo y criterios temporales.")
p("Separamos los parámetros que pueden cambiar dentro de un ejercicio de los que "
  "solo deben cambiar entre ejercicios, para mantener comparables las evaluaciones. "
  "La política usa suavizado, histéresis, persistencia y periodo refractario.")
p("Con 5,369 estimaciones de 15 participantes, estos mecanismos redujeron las "
  "acciones de dificultad de 7.21 a 0.61 por minuto &mdash; de una cada ocho "
  "segundos a una cada cien &mdash; sin reducir el acuerdo con la condición "
  "impuesta.")
p("Encontramos una asimetría importante: el sistema detecta bien estados similares "
  "a reposo pero identifica mal la alta demanda. Al alimentar la misma política con "
  "una entrada perfecta, el recall de alta demanda subió de 0.37 a 1.00, lo que "
  "demuestra que la limitación está en la percepción EEG y no en el controlador.")
p("La conclusión es que las dos direcciones de adaptación deberían usar fuentes "
  "distintas: EEG para aumentar el desafío cuando hay baja carga, y telemetría de "
  "desempeño para reducir la dificultad cuando aparecen errores. La siguiente etapa "
  "es la evaluación con usuarios.")

h1("Límites que el artículo declara")
bullets(["Ninguna persona ha usado todavía el visor con el lazo activo: se "
         "caracterizó el <i>controlador</i>, no la intervención.",
         "La rampa de demanda es impuesta, no observada.",
         "Los datos vienen de tareas cognitivas de escritorio, no de tareas motoras "
         "quirúrgicas.",
         "Los extremos de dificultad son decisiones razonadas, no calibradas con "
         "usuarios.",
         "El lazo háptico recalcula fuerzas a 50 Hz, por debajo del kilohertz "
         "habitual.",
         "La dificultad es un único eje: alguien puede hallar difícil la precisión "
         "y fácil la presión de tiempo."])


h1("Anexo: qué es la «condición verdadera»")
p("Esta distinción es importante para leer bien los resultados, y el artículo la "
  "declara explícitamente.")
p("Los datos vienen de un experimento controlado. Cada participante pasó por cuatro "
  "bloques y <b>la etiqueta dice en qué bloque estaba</b>, no qué le pasaba por la "
  "cabeza. De hecho la etiqueta está codificada en el nombre del archivo crudo:")
table([["Archivo", "Etiqueta", "Qué hacía el participante"],
       ["natural-N",   "0", "Sentado en reposo, sin tarea"],
       ["lowlevel-N",  "1", "La tarea en su versión fácil"],
       ["midlevel-N",  "2", "La tarea en dificultad media"],
       ["highlevel-N", "3", "La versión más exigente: más presión de tiempo y conflicto"]],
      [3.6*cm, 2.0*cm, 9.6*cm])
p("Es decir, la condición verdadera es <b>el protocolo experimental</b>: lo que el "
  "investigador le pidió hacer a esa persona en ese momento.")
E.append(Table([[Paragraph(
    "No es una medición del estado mental real. Alguien en el bloque «alta» puede "
    "estar relajado, y alguien en «reposo» puede estar ansioso. Por eso el artículo "
    "habla de estados <i>tipo reposo</i> y de condiciones <i>de alta demanda</i>, y "
    "nunca de un aprendiz «desenganchado» o «saturado».", S["cell"])]],
    colWidths=[15.2*cm],
    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND),
                      ("LEFTPADDING", (0, 0), (-1, -1), 11),
                      ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                      ("TOPPADDING", (0, 0), (-1, -1), 9),
                      ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                      ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACC)])))
gap(10)
p("En la prueba del sensor perfecto (resultado 3) esa etiqueta es justamente lo que "
  "se le dio al controlador en vez del score EEG: si la ventana venía del bloque de "
  "reposo se le pasó un 0, y si venía del bloque más exigente, un 3.")



# =============================================================== armado
def chrome(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5)
    canvas.line(2.4*cm, 1.55*cm, letter[0]-2.4*cm, 1.55*cm)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(2.4*cm, 1.1*cm, "Resumen ejecutivo · COMRob 2026")
    canvas.drawRightString(letter[0]-2.4*cm, 1.1*cm, f"{doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(str(OUT), pagesize=letter,
                      leftMargin=2.4*cm, rightMargin=2.4*cm,
                      topMargin=2.1*cm, bottomMargin=2.1*cm,
                      title="Resumen ejecutivo — COMRob 2026",
                      author="")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=chrome)])
doc.build(E)
print(f"[pdf] {OUT}")
