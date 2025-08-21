# app.py — InsightFutsal (Estadísticas de Partido desde data/minutos, sin uploader)
import os
import re
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from lxml import etree
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")

# Banner si existe
def _try_show_banner():
    banner_dir = "images/banner"
    if os.path.isdir(banner_dir):
        # busca jpg/png/webp en orden
        candidates = []
        for ext in ("*.png","*.jpg","*.jpeg","*.webp"):
            candidates += glob.glob(os.path.join(banner_dir, ext))
        if candidates:
            try:
                img = Image.open(candidates[0])
                st.image(img, use_container_width=True)
            except Exception:
                pass

_try_show_banner()
st.title("InsightFutsal — Estadísticas de Partido")

# =========================
# AJUSTES DE ETIQUETAS EXACTAS (CAMBIAR AQUÍ SI TU NOTEBOOK USA OTRAS)
# =========================
# Códigos para posesión (en 'code')
CODE_POSESION_FERRO = "Tiempo Posecion Ferro"
CODE_POSESION_RIVAL = "Tiempo Posecion Rival"

# Labels (en 'labels') para detección de eventos
LABEL_TIRO_1 = "Finalizacion jugada cTiro"     # tiro propio estándar
LABEL_TIRO_2 = "sTiro"                          # si usás sTiro en algunos
LABEL_GOL_FAVOR = "Goles a favor en cancha"
LABEL_GOL_RIVAL = "Gol Rival en cancha"

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def list_xml_matches(base_dir: str = "data/minutos") -> list[str]:
    """Busca archivos con 'TotalValues.xml' dentro de data/minutos."""
    if not os.path.isdir(base_dir):
        return []
    files = [f for f in os.listdir(base_dir)
             if f.lower().endswith(".xml") and "totalvalues" in f.lower()]
    files.sort()
    return files

def _labels_text(labels):
    if isinstance(labels, list):
        return " | ".join([str(x) for x in labels]).lower()
    return ""

def _has_label(labels, needle: str) -> bool:
    return needle.lower() in _labels_text(labels)

@st.cache_data(show_spinner=False)
def parse_asxml_totalvalues(xml_path: str) -> pd.DataFrame:
    """
    Parser robusto para NacSport asXML / asXML TotalValues.
    Devuelve DataFrame con: start, end, duration, code, labels(list), pos_x, pos_y.
    """
    try:
        with open(xml_path, "rb") as fh:
            xml_bytes = fh.read()
        root = etree.fromstring(xml_bytes)
        rows = []

        # Dos variantes comunes:
        instances = root.findall(".//ALL_INSTANCES/instance")
        if not instances:
            instances = root.findall(".//instance")

        for inst in instances:
            start = inst.findtext("start")
            end = inst.findtext("end")
            code = inst.findtext("code") or ""
            px = inst.findtext("pos_x")
            py = inst.findtext("pos_y")

            # labels: pueden venir como <label><text>..</text></label>
            labels = [t.text for t in inst.findall(".//label/text") if t is not None and t.text]

            row = {
                "start": float(start) if start else np.nan,
                "end": float(end) if end else np.nan,
                "duration": (float(end) - float(start)) if (start and end) else np.nan,
                "code": code,
                "pos_x": float(px) if px else np.nan,
                "pos_y": float(py) if py else np.nan,
                "labels": labels,
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"No pude parsear '{xml_path}': {e}")
        return pd.DataFrame(columns=["start","end","duration","code","pos_x","pos_y","labels"])

@st.cache_data(show_spinner=False)
def compute_match_stats(df: pd.DataFrame) -> dict:
    """
    Calcula: posesión propia/rival (por duración), tiros propios/rival, goles favor/rival.
    Ajustado a las etiquetas configuradas arriba.
    """
    if df.empty:
        return dict(
            posesion_ferro_pct=0.0,
            posesion_rival_pct=0.0,
            tiros_ferro=0, tiros_rival=0,
            goles_favor=0, goles_rival=0,
        )

    # Posesión por duración de instancias
    pos_ferro = df.loc[df["code"].str.contains(CODE_POSESION_FERRO, case=False, na=False), "duration"].sum()
    pos_rival = df.loc[df["code"].str.contains(CODE_POSESION_RIVAL, case=False, na=False), "duration"].sum()
    total = pos_ferro + pos_rival
    pf = (pos_ferro / total * 100) if total > 0 else 0.0
    pr = 100 - pf if total > 0 else 0.0

    # Tiros: buscamos por labels, y tratamos de asignarlos a lado Ferro/Rival
    tiro_mask = df["labels"].apply(lambda L:
        _has_label(L, LABEL_TIRO_1) or _has_label(L, LABEL_TIRO_2) or _has_label(L, " tiro")
    )
    # Heurística de lado: tomamos la intersección con bloques de posesión
    tiros_ferro = int( (df[tiro_mask & df["code"].str.contains(CODE_POSESION_FERRO, case=False, na=False)]).shape[0] )
    tiros_rival = int( (df[tiro_mask & df["code"].str.contains(CODE_POSESION_RIVAL, case=False, na=False)]).shape[0] )

    # Goles: deduplicamos por (start,end) por si el mismo gol aparece replicado
    gf_times = set(df.loc[df["labels"].apply(lambda L: _has_label(L, LABEL_GOL_FAVOR)),
                          ["start","end"]].itertuples(index=False, name=None))
    gr_times = set(df.loc[df["labels"].apply(lambda L: _has_label(L, LABEL_GOL_RIVAL)),
                          ["start","end"]].itertuples(index=False, name=None))
    goles_favor = len(gf_times)
    goles_rival = len(gr_times)

    return dict(
        posesion_ferro_pct=round(pf,1),
        posesion_rival_pct=round(pr,1),
        tiros_ferro=tiros_ferro,
        tiros_rival=tiros_rival,
        goles_favor=goles_favor,
        goles_rival=goles_rival,
    )

def extract_rival_from_filename(fname: str) -> str:
    """
    Extrae el rival desde 'Fecha N° - Rival - XML TotalValues.xml'
    """
    rival = re.sub(r"(?i)^fecha\s*\d+\s*-\s*", "", fname)  # quita 'Fecha N - '
    rival = re.sub(r"(?i)\s*-\s*xml\s*totalvalues\.xml$", "", rival)  # quita sufijo
    return rival.strip()

def try_team_badge(team_name: str):
    """
    Muestra escudo del equipo si existe en images/equipos/<team>.(png|jpg|webp)
    """
    base = "images/equipos"
    for ext in ("png","jpg","jpeg","webp"):
        p = os.path.join(base, f"{team_name}.{ext}")
        if os.path.isfile(p):
            try:
                return Image.open(p)
            except Exception:
                return None
    return None

# =========================
# LAYOUT
# =========================
with st.sidebar:
    st.header("Selección de partido")
    base_dir = "data/minutos"
    files = list_xml_matches(base_dir)
    if not files:
        st.error(f"No hay XML con 'TotalValues' en {base_dir}. Subí tus partidos allí.")
        st.stop()
    match = st.selectbox("Partido", files, index=0)
    rival_name = extract_rival_from_filename(match)

# Encabezado con escudos si existen
colA, colB, colC = st.columns([1,2,1])
with colA:
    ferro_badge = try_team_badge("Ferro")
    if ferro_badge: st.image(ferro_badge, caption="Ferro", use_container_width=True)
with colB:
    st.subheader(f"Fecha / Rival: {rival_name}")
with colC:
    rival_badge = try_team_badge(rival_name)
    if rival_badge: st.image(rival_badge, caption=rival_name, use_container_width=True)

# =========================
# CÁLCULO
# =========================
xml_path = os.path.join(base_dir, match)
df = parse_asxml_totalvalues(xml_path)

if df.empty:
    st.warning("No se pudo leer el XML o no tiene instancias.")
    st.stop()

stats = compute_match_stats(df)

# =========================
# MÉTRICAS + GRÁFICOS
# =========================
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

# Barras tiros & goles
bars_df = pd.DataFrame({
    "Equipo": ["Ferro", rival_name],
    "Tiros": [stats["tiros_ferro"], stats["tiros_rival"]],
    "Goles": [stats["goles_favor"], stats["goles_rival"]],
})
fig_bar_tiros = px.bar(bars_df, x="Equipo", y="Tiros", text="Tiros", title="Tiros por equipo")
fig_bar_goles = px.bar(bars_df, x="Equipo", y="Goles", text="Goles", title="Goles por equipo")
st.plotly_chart(fig_bar_tiros, use_container_width=True)
st.plotly_chart(fig_bar_goles, use_container_width=True)

# Debug opcional
with st.expander("Ver tabla base (debug)"):
    st.dataframe(df.head(300), use_container_width=True)

st.caption("Ajustá las etiquetas en la sección 'AJUSTES DE ETIQUETAS EXACTAS' si tu notebook usa strings distintas.")
