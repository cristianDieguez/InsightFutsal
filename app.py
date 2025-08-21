# app.py — InsightFutsal (MVP multipágina)
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")
st.title("InsightFutsal")

# =========================
# HELPERS (mínimos; los podrás reemplazar por tus funciones reales)
# =========================
@st.cache_data(show_spinner=False)
def parse_nacsport_xml(file_bytes: bytes) -> pd.DataFrame:
    """
    Parser mínimo de XML de NacSport -> DataFrame.
    Devuelve: start, end, code, pos_x, pos_y, labels(list), dur.
    Reemplazá por tu parser cuando quieras.
    """
    try:
        root = ET.fromstring(file_bytes)
        rows = []
        for inst in root.findall(".//instance"):
            start = inst.findtext("start")
            end = inst.findtext("end")
            code = inst.findtext("code")
            px = inst.findtext("pos_x")
            py = inst.findtext("pos_y")
            labels = [l.findtext("text") for l in inst.findall("label")]
            rows.append({
                "start": float(start) if start else np.nan,
                "end": float(end) if end else np.nan,
                "code": code or "",
                "pos_x": float(px) if px else np.nan,
                "pos_y": float(py) if py else np.nan,
                "labels": labels
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["dur"] = df["end"] - df["start"]
        return df
    except Exception as e:
        st.error(f"No pude leer el XML. Detalle: {e}")
        return pd.DataFrame(columns=["start","end","code","pos_x","pos_y","labels","dur"])

def _labels_text(labels):
    return " | ".join(labels).lower() if isinstance(labels, list) else ""

def is_shot(labels):  # ajustá a tus etiquetas reales
    t = _labels_text(labels)
    return ("tiro" in t) or ("shot" in t)

def is_goal(labels):
    t = _labels_text(labels)
    return ("gol" in t) and ("en contra" not in t)

@st.cache_data(show_spinner=False)
def minutes_by_code(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return pd.DataFrame(columns=["code","minutos"])
    out = df.groupby("code", dropna=False)["dur"].sum().reset_index()
    out["minutos"] = out["dur"].fillna(0) / 60.0
    return out.sort_values("minutos", ascending=False)[["code","minutos"]]

@st.cache_data(show_spinner=False)
def shots_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    s = df[df["labels"].apply(is_shot)].copy()
    if s.empty: return s
    s["is_goal"] = s["labels"].apply(is_goal)
    return s[["start","code","pos_x","pos_y","is_goal"]]

@st.cache_data(show_spinner=False)
def grid3_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conteos por celda 3x3 usando pos_x/pos_y normalizados.
    Ajustá a tu sistema (35x20) cuando conectemos tu lógica real.
    """
    tmp = df.dropna(subset=["pos_x","pos_y"]).copy()
    if tmp.empty: 
        return pd.DataFrame(columns=["gx","gy","n"])
    # Normalizamos a 0..1 para bucketing
    tmp["nx"] = (tmp["pos_x"] - tmp["pos_x"].min()) / (tmp["pos_x"].max() - tmp["pos_x"].min() + 1e-9)
    tmp["ny"] = (tmp["pos_y"] - tmp["pos_y"].min()) / (tmp["pos_y"].max() - tmp["pos_y"].min() + 1e-9)
    tmp["gx"] = (tmp["nx"] * 3).clip(0, 2.9999).astype(int)
    tmp["gy"] = (tmp["ny"] * 3).clip(0, 2.9999).astype(int)
    return tmp.groupby(["gy","gx"]).size().reset_index(name="n").sort_values("n", ascending=False)

# =========================
# SIDEBAR — menú y carga
# =========================
st.sidebar.header("⚙️ Opciones")
page = st.sidebar.radio(
    "Menú",
    ["📦 Resumen", "⏱️ Minutos", "🎯 Tiros", "🗺️ Mapa 3x3", "📊 Estadísticas"]
)

st.sidebar.markdown("---")
xml_file = st.sidebar.file_uploader("Subí XML (NacSport)", type=["xml"])
show_debug = st.sidebar.checkbox("Mostrar tablas (debug)")

# Si no hay archivo, mostramos aviso en todas las páginas
if xml_file is None:
    st.info("Subí un XML para comenzar.")
    st.stop()

# Parseo (cacheado)
df = parse_nacsport_xml(xml_file.read())

# =========================
# PÁGINAS
# =========================
if page == "📦 Resumen":
    st.subheader("Resumen del archivo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Instancias", len(df))
    c2.metric("Con coordenadas", int(df.dropna(subset=["pos_x","pos_y"]).shape[0]))
    c3.metric("Con duración", int(df.dropna(subset=["dur"]).shape[0]))
    if show_debug:
        st.dataframe(df.head(30), use_container_width=True)

elif page == "⏱️ Minutos":
    st.subheader("Minutos por 'code' (Jugador/Rol)")
    mins = minutes_by_code(df)
    if mins.empty:
        st.info("No hay datos de duración para calcular minutos.")
    else:
        c1, c2 = st.columns([1,2])
        with c1:
            st.dataframe(mins, use_container_width=True)
        with c2:
            fig = px.bar(
                mins.head(25),
                x="minutos", y="code", orientation="h",
                text=mins.head(25)["minutos"].round(1),
                title="Top 25 — Minutos"
            )
            fig.update_layout(xaxis_title="Minutos", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Tiros":
    st.subheader("Tiros (ubicación y goles)")
    shots = shots_table(df)
    if shots.empty:
        st.info("No se detectaron tiros según las etiquetas.")
    else:
        if show_debug:
            st.dataframe(shots.head(30), use_container_width=True)
        fig = px.scatter(
            shots, x="pos_x", y="pos_y",
            color=shots["is_goal"].map({True: "Gol", False: "Sin gol"}),
            hover_data=["code","start"],
            title="Ubicación de tiros (color: gol)"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🗺️ Mapa 3x3":
    st.subheader("Mapa 3x3 — Conteos por celda")
    g = grid3_counts(df)
    if g.empty:
        st.info("No hay posiciones para mapear.")
    else:
        mat = np.zeros((3,3))
        for _, r in g.iterrows():
            mat[int(r["gy"]), int(r["gx"])] = r["n"]
        heat = pd.DataFrame(mat, columns=[0,1,2], index=[0,1,2])
        fig = px.imshow(heat, text_auto=True, title="Conteos por celda (3x3)")
        st.plotly_chart(fig, use_container_width=True)
        if show_debug:
            st.dataframe(g, use_container_width=True)

elif page == "📊 Estadísticas":
    st.subheader("Estadísticas simples (MVP)")
    # Ejemplos básicos: total instancias, % con coords, % tiros/goles
    total = len(df)
    with_coords = int(df.dropna(subset=["pos_x","pos_y"]).shape[0])
    shots = shots_table(df)
    goals = shots["is_goal"].sum() if not shots.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instancias", total)
    c2.metric("% con coords", f"{(with_coords/total*100):.1f}%" if total else "0%")
    c3.metric("Tiros detectados", 0 if shots.empty else len(shots))
    c4.metric("Goles detectados", int(goals))
    st.caption("Esta página es solo un placeholder. Vamos a conectar tus métricas reales (xG, posesión, etc.).")
