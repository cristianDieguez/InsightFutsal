# app.py — InsightFutsal (MVP multipágina)
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import os

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

@st.cache_data(show_spinner=False)
def parse_asxml_file(file_path: str) -> pd.DataFrame:
    """
    Lee un XML (NacSport asXML/TotalValues) desde disco y devuelve un DataFrame
    con columnas: start, end, duration, code, labels(list), pos_x, pos_y.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        rows = []
        # intentamos ambos paths comunes de NacSport
        instances = root.findall(".//ALL_INSTANCES/instance")
        if not instances:
            instances = root.findall(".//instance")
        for inst in instances:
            start = inst.findtext("start")
            end = inst.findtext("end")
            code = inst.findtext("code") or ""
            px = inst.findtext("pos_x")
            py = inst.findtext("pos_y")
            labels = [l.findtext("text") for l in inst.findall(".//label") if l.findtext("text")]
            rows.append({
                "start": float(start) if start else np.nan,
                "end": float(end) if end else np.nan,
                "duration": (float(end) - float(start)) if start and end else np.nan,
                "code": code,
                "pos_x": float(px) if px else np.nan,
                "pos_y": float(py) if py else np.nan,
                "labels": labels
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error leyendo {file_path}: {e}")
        return pd.DataFrame(columns=["start","end","duration","code","pos_x","pos_y","labels"])

def _has_label(labels, needle: str) -> bool:
    if not isinstance(labels, list): return False
    t = " | ".join(labels).lower()
    return needle.lower() in t

@st.cache_data(show_spinner=False)
def compute_match_stats(df: pd.DataFrame) -> dict:
    """
    Ajustá las cadenas de 'code' y labels a tus etiquetas del notebook.
    """
    if df.empty:
        return dict(posesion_ferro_pct=0, posesion_rival_pct=0,
                    tiros_ferro=0, tiros_rival=0,
                    goles_favor=0, goles_rival=0)

    # POSESIÓN por duración (ajustá a tus 'code' exactos del notebook)
    pos_ferro = df.loc[df["code"].str.contains("Posecion Ferro", case=False, na=False), "duration"].sum()
    pos_rival = df.loc[df["code"].str.contains("Posecion Rival", case=False, na=False), "duration"].sum()
    total_pos = pos_ferro + pos_rival
    pf = (pos_ferro / total_pos * 100) if total_pos > 0 else 0
    pr = 100 - pf if total_pos > 0 else 0

    # TIROS (ajustá el texto del label a lo que uses en NacSport)
    tiro_mask = df["labels"].apply(lambda L: _has_label(L, "Finalizacion jugada cTiro") or _has_label(L, "tiro"))
    tiros_ferro = df[tiro_mask & df["code"].str.contains("Ferro", case=False, na=False)].shape[0]
    tiros_rival = df[tiro_mask & df["code"].str.contains("Rival", case=False, na=False)].shape[0]

    # GOLES (deduplico por (start,end) por si el gol aparece en varias instancias)
    goles_favor_times = set(df.loc[df["labels"].apply(lambda L: _has_label(L, "Goles a favor")), ["start","end"]].itertuples(index=False, name=None))
    goles_rival_times = set(df.loc[df["labels"].apply(lambda L: _has_label(L, "Gol Rival")), ["start","end"]].itertuples(index=False, name=None))

    return dict(
        posesion_ferro_pct=round(pf,1),
        posesion_rival_pct=round(pr,1),
        tiros_ferro=int(tiros_ferro),
        tiros_rival=int(tiros_rival),
        goles_favor=len(goles_favor_times),
        goles_rival=len(goles_rival_times),
    )

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
    st.subheader("Estadísticas de partido (desde data/minutos)")
    base_dir = "data/minutos"
    if not os.path.isdir(base_dir):
        st.warning(f"No existe la carpeta {base_dir} en el repo.")
    else:
        files = sorted([f for f in os.listdir(base_dir) if f.lower().endswith(".xml")])
        if not files:
            st.info(f"No hay XML en {base_dir}. Subí tus archivos ahí.")
        else:
            col1, col2 = st.columns([2,1])
            with col1:
                file_selected = st.selectbox("Elegí el partido (XML)", files, index=0)
            with col2:
                rival_name = st.text_input("Nombre del rival (solo etiqueta)", value="Rival")

            xml_path = os.path.join(base_dir, file_selected)
            df_match = parse_asxml_file(xml_path)
            if df_match.empty:
                st.warning("No pude parsear el XML o está vacío.")
            else:
                stats = compute_match_stats(df_match)

                c1,c2,c3,c4,c5,c6 = st.columns(6)
                c1.metric("Posesión propia", f"{stats['posesion_ferro_pct']}%")
                c2.metric("Posesión rival", f"{stats['posesion_rival_pct']}%")
                c3.metric("Tiros propios", stats["tiros_ferro"])
                c4.metric("Tiros rival", stats["tiros_rival"])
                c5.metric("Goles a favor", stats["goles_favor"])
                c6.metric("Goles en contra", stats["goles_rival"])

                # Pie de posesión
                pie_df = pd.DataFrame({
                    "Equipo": [f"Ferro vs {rival_name}", rival_name],
                    "Posesión": [stats['posesion_ferro_pct'], stats['posesion_rival_pct']]
                })
                fig_pie = px.pie(pie_df, names="Equipo", values="Posesión", title="Posesión (%)")
                st.plotly_chart(fig_pie, use_container_width=True)

                if show_debug:
                    st.subheader("Tabla base (debug)")
                    st.dataframe(df_match.head(200), use_container_width=True)

    st.caption("Ajustamos etiquetas de 'code' y 'labels' a las que uses en tu NacSport/Notebook.")

