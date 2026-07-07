# Diego Sanjur 8-1024-2362
# Jorge Valderrama 8-1023-157
# Carlos Reyes 8-849-624
# Aaron Burac 8-1049-1605

import webbrowser
import threading
import numpy as np
from scipy import stats
from flask import Flask
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

import backend

# Paleta de colores y estilo base
VERDE = "#2f9e44"
VERDE_OSCURO = "#1f3b2c"
NARANJA = "#e8590c"
NARANJA_CLARO = "#ffa94d"
GRIS = "#495057"

FUENTE = "Segoe UI, Roboto, Helvetica, Arial, sans-serif"

#Nombres de variables
NOMBRES = {
    'FFMC': 'FFMC (humedad superficial)',
    'DMC': 'DMC (humedad media del suelo)',
    'DC': 'DC (sequía profunda)',
    'ISI': 'ISI (propagación del fuego)',
    'temp': 'Temperatura (°C)',
    'RH': 'Humedad relativa (%)',
    'wind': 'Viento (km/h)',
    'rain': 'Lluvia (mm)',
    'area': 'Área quemada (ha)',
}

MESES_ORDEN = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
MESES_ES = {'jan': 'Ene', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Abr',
            'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Ago',
            'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dic'}
DIAS_ORDEN = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
DIAS_ES = {'mon': 'Lun', 'tue': 'Mar', 'wed': 'Mié', 'thu': 'Jue',
           'fri': 'Vie', 'sat': 'Sáb', 'sun': 'Dom'}

# Preparacion de datos
RUTA_CSV = 'forestfires.csv'

df_raw = backend.cargar_datos(RUTA_CSV)                      # datos original
df_nulos, mascara_nulos = backend.introducir_nulos(df_raw)  # ~10% nulos en temp
df_limpio = backend.imputar_knn(df_nulos, n_vecinos=5)       # imputados (analisis)

VARIABLES_NUMERICAS = backend.COLUMNAS_NUMERICAS
N_NULOS = int(mascara_nulos.sum())
PCT_NULOS = 100 * N_NULOS / len(df_raw)


def estilizar(fig, titulo):
    """Aplica un look consistente a cualquier figura."""
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16, color=VERDE_OSCURO)),
        template="plotly_white",
        font=dict(family=FUENTE, color=GRIS),
        margin=dict(l=55, r=25, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig

# Graficas estaticas
def figura_correlacion():
    corr = df_limpio[VARIABLES_NUMERICAS].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(title="r"),
    ))
    return estilizar(fig, "Correlación entre variables meteorológicas y de fuego")

def figura_boxplot():
    #z-score para poder comparar rangos y outliers en una misma escala
    z = (df_limpio[VARIABLES_NUMERICAS] - df_limpio[VARIABLES_NUMERICAS].mean()) \
        / df_limpio[VARIABLES_NUMERICAS].std()
    fig = go.Figure()
    for col in VARIABLES_NUMERICAS:
        fig.add_trace(go.Box(y=z[col], name=col, marker_color=VERDE,
                             line_color=VERDE_OSCURO, boxpoints="outliers"))
    fig.update_layout(showlegend=False, yaxis_title="Valor estandarizado (z)")
    return estilizar(fig, "Rango y casos extremos de cada variable (estandarizado)")

# Graficas dinamicas
def figura_distribucion(col):
    serie = df_limpio[col].dropna()
    mu, sigma = serie.mean(), serie.std()
    xs = np.linspace(serie.min(), serie.max(), 200)
    ys = stats.norm.pdf(xs, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=serie, histnorm="probability density", nbinsx=30,
        name="Datos reales", marker_color=VERDE, opacity=0.75))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="Curva normal ideal",
        line=dict(color=NARANJA, width=3)))
    fig.update_layout(xaxis_title=NOMBRES[col], yaxis_title="Densidad", bargap=0.05)
    return estilizar(fig, f"Distribución de {NOMBRES[col]}")

def figura_normalidad(col):
    serie = df_limpio[col].dropna()
    (osm, osr), (pend, inter, r) = stats.probplot(serie, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers", name="Datos reales",
        marker=dict(color=VERDE, size=6, opacity=0.7)))
    fig.add_trace(go.Scatter(
        x=osm, y=pend * osm + inter, mode="lines", name="Si fuera normal",
        line=dict(color=NARANJA, width=2)))
    fig.update_layout(xaxis_title="Cuantiles teóricos (normal)",
                      yaxis_title="Cuantiles reales de la muestra")
    return estilizar(fig, f"¿Es normal la variable {NOMBRES[col]}? (Q-Q)")

def texto_shapiro(col):
    serie = df_limpio[col].dropna()
    est, p = stats.shapiro(serie)
    nombre = NOMBRES[col]
    if p < 0.05:
        veredicto = f"{nombre} NO sigue una distribución normal"
        color = NARANJA
        signo = "< 0.05"
    else:
        veredicto = f"{nombre} es compatible con una distribución normal"
        color = VERDE
        signo = "≥ 0.05"
    return html.Span([
        html.B("Prueba de Shapiro-Wilk: "),
        f"W = {est:.3f}, p = {p:.4g}. Como p {signo}, ",
        html.Span(veredicto, style={"color": color, "fontWeight": 600}),
        ".",
    ])

def figura_agrupacion(categoria, metrica):
    orden = MESES_ORDEN if categoria == "month" else DIAS_ORDEN
    etiquetas = MESES_ES if categoria == "month" else DIAS_ES
    if metrica == "conteo":
        serie = df_limpio.groupby(categoria).size()
        etiqueta = "Cantidad de incendios"
    elif metrica == "area_prom":
        serie = df_limpio.groupby(categoria)["area"].mean()
        etiqueta = "Área quemada promedio (ha)"
    else:  # area_total
        serie = df_limpio.groupby(categoria)["area"].sum()
        etiqueta = "Área quemada total (ha)"

    serie = serie.reindex(orden)
    fig = go.Figure(go.Bar(
        x=[etiquetas[k] for k in serie.index], y=serie.values,
        marker_color=NARANJA, marker_line_color=NARANJA, opacity=0.9))
    nombre_cat = "mes" if categoria == "month" else "día de la semana"
    fig.update_layout(xaxis_title=nombre_cat.capitalize(), yaxis_title=etiqueta)
    return estilizar(fig, f"{etiqueta} por {nombre_cat}")


def figura_imputacion(n_vecinos):
    df_imp = backend.imputar_knn(df_nulos, n_vecinos=n_vecinos)
    reales = df_raw.loc[mascara_nulos, "temp"].values
    imputados = df_imp.loc[mascara_nulos, "temp"].values
    mae = np.mean(np.abs(reales - imputados))

    lim_min = min(reales.min(), imputados.min()) - 1
    lim_max = max(reales.max(), imputados.max()) + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lim_min, lim_max], y=[lim_min, lim_max], mode="lines",
        name="Predicción perfecta", line=dict(color=GRIS, dash="dash")))
    fig.add_trace(go.Scatter(
        x=reales, y=imputados, mode="markers", name="Temperaturas estimadas",
        marker=dict(color=VERDE, size=9, opacity=0.75,
                    line=dict(color=VERDE_OSCURO, width=1))))
    fig.update_layout(xaxis_title="Temperatura real (°C)",
                      yaxis_title="Temperatura estimada por KNN (°C)")
    fig = estilizar(fig, f"¿Qué tan bien adivinó el modelo? (k = {n_vecinos})")

    texto = html.Span([
        "Se estimaron ", html.B(f"{N_NULOS} temperaturas"), " ocultas. En promedio, "
        "el modelo se equivocó por ",
        html.B(f"{mae:.2f} °C", style={"color": NARANJA}),
        " (error absoluto medio). Mientras menor sea este número, mejor la imputación.",
    ])
    return fig, texto

# App: Flask como servidor y Dash montado encima
servidor = Flask(__name__)
app = Dash(__name__, server=servidor, title="Tarea 3 - Forest Fires")


def tarjeta(valor, etiqueta):
    return html.Div(className="kpi", children=[
        html.Div(valor, className="kpi-valor"),
        html.Div(etiqueta, className="kpi-etiqueta"),
    ])


def desc(texto): # Parrafo explicativo que se muestra bajo el titulo de cada seccion.
    return html.P(texto, className="desc")

def panel(titulo, hijos):
    return html.Div(className="panel", children=[
        html.H3(titulo, className="panel-titulo"),
        *hijos,
    ])

app.layout = html.Div(className="contenedor", children=[

    html.Header(className="cabecera", children=[
        html.H1("Análisis de incendios forestales · Parque Montesinho")
    ]),

    # Tarjetas resumen
    html.Div(className="kpis", children=[
        tarjeta(f"{len(df_raw)}", "Incendios registrados"),
        tarjeta(f"{len(VARIABLES_NUMERICAS)}", "Variables numéricas"),
        tarjeta(f"{N_NULOS} ({PCT_NULOS:.1f}%)", "Nulos imputados en temperatura"),
        tarjeta(f"{df_limpio['area'].sum():,.0f} ha", "Área total quemada"),
    ]),

    # Fila 1: Distribucion + normalidad
    panel("Distribución y normalidad de una variable", [
        desc("Cómo se reparten los valores de la variable (izquierda) y si sigue una distribución normal (derecha)."),
        html.Div(className="control", children=[
            html.Label("Variable a analizar:"),
            dcc.Dropdown(
                id="dd-variable",
                options=[{"label": NOMBRES[v], "value": v} for v in VARIABLES_NUMERICAS],
                value="temp", clearable=False, style={"width": "340px"}),
        ]),
        html.Div(className="grid-2", children=[
            dcc.Graph(id="graf-distribucion"),
            dcc.Graph(id="graf-normalidad"),
        ]),
        html.Div(id="texto-shapiro", className="nota"),
    ]),

    # Fila 2: agrupacion categorica (callback 2, dos controles)
    panel("Estacionalidad - ¿Cuándo ocurren los incendios?", [
        desc("Incendios agrupados por mes o día. Conteo = cuántos hubo; Área = qué tan grandes fueron."),
        html.Div(className="controles", children=[
            html.Div(className="control", children=[
                html.Label("Agrupar por:"),
                dcc.RadioItems(
                    id="radio-categoria",
                    options=[{"label": " Mes", "value": "month"},
                             {"label": " Día de la semana", "value": "day"}],
                    value="month", inline=True),
            ]),
            html.Div(className="control", children=[
                html.Label("Métrica:"),
                dcc.RadioItems(
                    id="radio-metrica",
                    options=[{"label": " Conteo de incendios", "value": "conteo"},
                             {"label": " Área promedio", "value": "area_prom"},
                             {"label": " Área total", "value": "area_total"}],
                    value="conteo", inline=True),
            ]),
        ]),
        dcc.Graph(id="graf-agrupacion"),
    ]),

    # Fila 3: correlacion + boxplot (estaticas)
    html.Div(className="grid-2", children=[
        panel("¿Qué variables se relacionan entre sí?", [
            desc("De -1 a 1: verde = suben juntas, rojo = una sube y otra baja, cerca "
                 "de 0 = sin relación."),
            dcc.Graph(figure=figura_correlacion()),
        ]),
        panel("Rangos y casos extremos", [
            desc("Cada caja resume una variable (mediana, 50% central). Los puntos sueltos son atípicos. Estandarizadas para compararlas."),
            dcc.Graph(figure=figura_boxplot()),
        ]),
    ]),

    # Fila 4: calidad de imputacion (callback 3, slider)
    panel("Calidad de la imputación KNN", [
        desc("Se ocultó el 13% de las temperaturas y el modelo las estimó. Cada punto: eje X = real, eje Y = estimado."),
        html.Div(className="control", children=[
            html.Label("Número de vecinos (k):"),
            dcc.Slider(id="slider-vecinos", min=1, max=15, step=1, value=5,
                       marks={i: str(i) for i in range(1, 16)}),
        ]),
        dcc.Graph(id="graf-imputacion"),
        html.Div(id="texto-imputacion", className="nota"),
    ])
])

# Callbacks
@app.callback(
    Output("graf-distribucion", "figure"),
    Output("graf-normalidad", "figure"),
    Output("texto-shapiro", "children"),
    Input("dd-variable", "value"),
)
def actualizar_variable(col):
    return figura_distribucion(col), figura_normalidad(col), texto_shapiro(col)


@app.callback(
    Output("graf-agrupacion", "figure"),
    Input("radio-categoria", "value"),
    Input("radio-metrica", "value"),
)
def actualizar_agrupacion(categoria, metrica):
    return figura_agrupacion(categoria, metrica)


@app.callback(
    Output("graf-imputacion", "figure"),
    Output("texto-imputacion", "children"),
    Input("slider-vecinos", "value"),
)
def actualizar_imputacion(n_vecinos):
    return figura_imputacion(n_vecinos)

# Estilos (CSS embebido para que el proyecto sea de un solo bloque portable)
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
        body { margin:0; background:#f1f4f1; font-family: """ + FUENTE + """; }
        .contenedor { max-width:1200px; margin:0 auto; padding:0 20px 40px; }
        .cabecera { background:""" + VERDE_OSCURO + """; color:#fff;
            padding:26px 30px; border-radius:0 0 14px 14px; margin-bottom:22px; }
        .cabecera h1 { margin:0 0 6px; font-size:24px; }
        .cabecera p { margin:0; opacity:.85; font-size:14px; }
        .kpis { display:grid; grid-template-columns:repeat(4,1fr);
            gap:16px; margin-bottom:22px; }
        .kpi { background:#fff; border-radius:12px; padding:18px 20px;
            border-left:5px solid """ + NARANJA + """;
            box-shadow:0 1px 4px rgba(0,0,0,.06); }
        .kpi-valor { font-size:24px; font-weight:700; color:""" + VERDE_OSCURO + """; }
        .kpi-etiqueta { font-size:13px; color:#6c757d; margin-top:2px; }
        .panel { background:#fff; border-radius:14px; padding:20px 22px;
            margin-bottom:22px; box-shadow:0 1px 4px rgba(0,0,0,.06); }
        .panel-titulo { margin:0 0 8px; color:""" + VERDE_OSCURO + """;
            font-size:18px; border-bottom:2px solid #edf2ed; padding-bottom:8px; }
        .desc { color:#5f6b62; font-size:13.5px; line-height:1.55;
            margin:0 0 16px; max-width:900px; }
        .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
        .controles { display:flex; flex-wrap:wrap; gap:32px; margin-bottom:8px; }
        .control { margin-bottom:12px; }
        .control label { font-weight:600; color:""" + GRIS + """;
            margin-right:10px; display:inline-block; margin-bottom:6px; }
        .nota { margin-top:10px; font-size:14px; color:""" + GRIS + """;
            background:#f7faf7; padding:10px 14px; border-radius:8px; }
        .pie { text-align:center; color:#868e96; font-size:12px; margin-top:10px; }
        @media (max-width:900px){
            .kpis{grid-template-columns:repeat(2,1fr);}
            .grid-2{grid-template-columns:1fr;}
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

def abrir_navegador():
    webbrowser.open_new("http://127.0.0.1:8050")

if __name__ == "__main__":
    # Abre el navegador automaticamente
    threading.Timer(1.2, abrir_navegador).start()
    app.run(debug=False, port=8050)