# app.py — InsightFutsal: Estadísticas de Partido (panel KEY STATS)
# Lee XML desde data/minutos/ y renderiza el panel estilo imagen.
# No usa uploader. Usa escudos de images/equipos y banner de images/banner.

import os, re, glob
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from lxml import etree

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")

DATA_DIR   = "data/minutos"
BADGE_DIR  = "images/equipos"
BANNER_DIR = "images/banner"

# ====== AJUSTES (CAMBIÁ SOLO ESTO SI TUS NOMBRES SON DISTINTOS) ======
# Códigos de 'code' para bloques de posesión (duración = posesión)
CODE_POSESION_FERRO = "Tiempo Posecion Ferro"
CODE_POSESION_RIVAL = "Tiempo Posecion Rival"

# Diccionario de métricas:
#   "Nombre mostrado": ([labels_ferro], [labels_rival], tipo, es_porcentaje)
# tipos soportados:
#   - "count"               => conteo simple por labels
#   - "count_unique_goal"   => goles (deduplicados por (start,end))
#   - "percent_possession"  => calculado aparte (no usa labels)
#   - "ratio_pases_ok"      => pases ok / pases totales
#   - "ratio_duelos"        => duelos ganados / total duelos
LABELS: Dict[str, Tuple[List[str], List[str], str, bool]] = {
    "Posesión %":            ([], [], "percent_possession", True),

    "Pases totales":         (["pases totales", "pase total"],
                              ["pases totales rival", "pase total rival"], "count", False),

    "Pases OK %":            (["pase ok"], ["pase ok rival"], "ratio_pases_ok", True),

    "Pases último tercio":   (["pase ultimo tercio"], ["pase ultimo tercio rival"], "count", False),
    "Pases al área":         (["pase al area"], ["pase al area rival"], "count", False),

    "Tiros":                 (["tiro", "finalizacion jugada ctiro", "stiro"],
                              ["tiro rival"], "count", False),
    "Tiros al arco":         (["tiro al arco"], ["tiro al arco rival"], "count", False),

    "Recuperaciones":        (["recuperacion"], ["recuperacion rival"], "count", False),

    "Duelos ganados":        (["duelo ganado"], ["duelo ganado rival"], "count", False),
    "% Duelos ganados":      (["duelo ganado"], ["duelo ganado rival"], "ratio_duelos", True),

    "Corners":               (["corner"], ["corner rival"], "count", False),
    "Faltas":                (["falta"], ["falta rival"], "count", False),

    "Goles":                 (["goles a favor en cancha", "gol a favor"],
                              ["gol rival en cancha", "gol rival"], "count_unique_goal", False),

    "Asistencias":           (["asistencia"], ["asistencia rival"], "count", False),
    "Pases clave":           (["pase clave"], ["pase clave rival"], "count", False),
}

# Estilo del panel
TITLE_HOME = "FERRO"
PANEL_BG   = "#0f5b33"
COLOR_HOME = "#ff8c2a"   # naranja propio
COLOR_AWAY = "#96a1a1"   # gris rival

# =========================================================
# UTILIDADES DE ARCHIVOS / IMÁGENES
# =========================================================
def list_xml_files(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    files = [f for f in os.listdir(base_dir)
             if f.lower().endswith(".xml") and "totalvalues" in f.lower()]
    files.sort()
    return files

def pretty_match_name(fname: str) -> str:
    """'Fecha 8 - Union Ezpeleta - XML TotalValues.xml' -> 'Fecha 8 – Union Ezpeleta'"""
    base = re.sub(r"(?i)\s*-\s*xml\s*totalvalues\.xml$", "", fname)
    return base

def extract_rival(fname: str) -> str:
    rival = re.sub(r"(?i)^fecha\s*\d+\s*-\s*", "", fname)
    rival = re.sub(r"(?i)\s*-\s*xml\s*totalvalues\.xml$", "", rival)
    return rival.strip()

def open_any(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None

def first_banner() -> Optional[Image.Image]:
    if not os.path.isdir(BANNER_DIR):
        return None
    for ext in ("png","jpg","jpeg","webp"):
        for p in glob.glob(os.path.join(BANNER_DIR, f"*.{ext}")):
            img = open_any(p)
            if img: return img
    return None

def badge_for(team: str) -> Optional[Image.Image]:
    if not os.path.isdir(BADGE_DIR):
        return None
    name = team.strip().lower()
    cands = []
    for ext in ("png","jpg","jpeg","webp"):
        cands += [
            os.path.join(BADGE_DIR, f"{name}.{ext}"),
            os.path.join(BADGE_DIR, f"{re.sub(r'\\s+', '_', name)}.{ext}"),
            os.path.join(BADGE_DIR, f"{re.sub(r'\\s+', '', name)}.{ext}")
        ]
    for p in cands:
        if os.path.isfile(p):
            img = open_any(p)
            if img: return img
    return None

# =========================================================
# PARSER asXML TotalValues → DataFrame
# =========================================================
def normtext(s: str) -> str:
    return (s or "").strip().lower()

def labels_text_list(node) -> List[str]:
    return [normtext(t.text) for t in node.findall(".//label/text") if t is not None and t.text]

@st.cache_data(show_spinner=False)
def parse_asxml(xml_path: str) -> pd.DataFrame:
    try:
        with open(xml_path, "rb") as fh:
            root = etree.fromstring(fh.read())
        inst = root.findall(".//ALL_INSTANCES/instance")
        if not inst: inst = root.findall(".//instance")

        rows = []
        for it in inst:
            start = it.findtext("start"); end = it.findtext("end")
            code  = normtext(it.findtext("code"))
            px = it.findtext("pos_x"); py = it.findtext("pos_y")
            labs = labels_text_list(it)
            rows.append({
                "start": float(start) if start else np.nan,
                "end":   float(end)   if end   else np.nan,
                "dur":   (float(end)-float(start)) if (start and end) else np.nan,
                "code":  code,
                "pos_x": float(px) if px else np.nan,
                "pos_y": float(py) if py else np.nan,
                "labels": labs
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error parseando {xml_path}: {e}")
        return pd.DataFrame(columns=["start","end","dur","code","pos_x","pos_y","labels"])

# =========================================================
# KPI CALCULADAS
# =========================================================
def _has_any(labels: List[str], needles: List[str]) -> bool:
    if not needles: return False
    tt = " | ".join(labels)
    return any(n.lower() in tt for n in needles)

def compute_possession(df: pd.DataFrame) -> Tuple[float,float]:
    own = df.loc[df["code"].str.contains(normtext(CODE_POSESION_FERRO), na=False), "dur"].sum()
    away = df.loc[df["code"].str.contains(normtext(CODE_POSESION_RIVAL), na=False), "dur"].sum()
    total = own + away
    if total <= 0: return 0.0, 0.0
    p_own = round(own/total*100, 1)
    p_away = round(100 - p_own, 1)
    return p_own, p_away

def count_by_labels(df: pd.DataFrame, keywords: List[str]) -> int:
    if not keywords: return 0
    return int(df["labels"].apply(lambda L: _has_any(L, keywords)).sum())

def count_unique_goal(df: pd.DataFrame, keywords: List[str]) -> int:
    sub = df[df["labels"].apply(lambda L: _has_any(L, keywords))][["start","end"]].dropna()
    if sub.empty: return 0
    pairs = set(sub.itertuples(index=False, name=None))
    return len(pairs)

def ratio(a:int, b:int) -> float:
    return round(a / b * 100.0, 1) if b > 0 else 0.0

def compute_kpis(df: pd.DataFrame) -> Dict[str, Tuple[float,float,bool]]:
    out: Dict[str, Tuple[float,float,bool]] = {}

    # Posesión
    own_pos, away_pos = compute_possession(df)
    out["Posesión %"] = (own_pos, away_pos, True)

    # Pases totales y OK %
    pases_home = count_by_labels(df, LABELS["Pases totales"][0])
    pases_away = count_by_labels(df, LABELS["Pases totales"][1])
    out["Pases totales"] = (pases_home, pases_away, False)

    pases_ok_home = count_by_labels(df, LABELS["Pases OK %"][0])
    pases_ok_away = count_by_labels(df, LABELS["Pases OK %"][1])
    out["Pases OK %"] = (ratio(pases_ok_home, max(pases_home,1)), ratio(pases_ok_away, max(pases_away,1)), True)

    # Conteos directos
    for key in ["Pases último tercio","Pases al área","Tiros","Tiros al arco",
                "Recuperaciones","Duelos ganados","Corners","Faltas","Asistencias","Pases clave"]:
        own = count_by_labels(df, LABELS[key][0])
        away = count_by_labels(df, LABELS[key][1])
        out[key] = (own, away, LABELS[key][3])

    # % Duelos ganados
    own_duelos, away_duelos, _ = out["Duelos ganados"]
    total_duelos = own_duelos + away_duelos
    own_duelos_pct = round(own_duelos/total_duelos*100,1) if total_duelos>0 else 0.0
    away_duelos_pct = round(100 - own_duelos_pct, 1) if total_duelos>0 else 0.0
    out["% Duelos ganados"] = (own_duelos_pct, away_duelos_pct, True)

    # Goles únicos
    own_goals = count_unique_goal(df, LABELS["Goles"][0])
    away_goals = count_unique_goal(df, LABELS["Goles"][1])
    out["Goles"] = (own_goals, away_goals, False)

    return out

# =========================================================
# RENDER DEL PANEL "KEY STATS" (Matplotlib)
# =========================================================
def draw_key_stats_panel(home_name: str, away_name: str,
                         kpis: Dict[str, Tuple[float,float,bool]],
                         home_badge: Optional[Image.Image],
                         away_badge: Optional[Image.Image]):
    order = [
        "Posesión %","Pases totales","Pases OK %","Pases último tercio","Pases al área",
        "Tiros","Tiros al arco","Recuperaciones","Duelos ganados","% Duelos ganados",
        "Corners","Faltas","Goles","Asistencias","Pases clave"
    ]

    n = len(order)
    h = 1.3 + n*0.43 + 1.1
    fig = plt.figure(figsize=(10.5, h), dpi=220)
    ax = plt.gca()
    ax.set_facecolor(PANEL_BG)
    fig.patch.set_facecolor(PANEL_BG)
    ax.axis("off")

    # Título
    title = f"{home_name.upper()} vs {away_name.upper()}"
    ax.text(0.5, 1.04, title, ha="center", va="top", color="white",
            fontsize=22, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 1.00, "KEY STATS", ha="center", va="top", color="white",
            fontsize=13, transform=ax.transAxes)

    # Escudos
    if home_badge:
        ax.imshow(home_badge, extent=[0.02, 0.12, 0.93, 1.07], aspect="auto")
    if away_badge:
        ax.imshow(away_badge, extent=[0.88, 0.98, 0.93, 1.07], aspect="auto")

    # Barras
    y0 = 0.92
    y_step = 0.043
    for i, key in enumerate(order):
        if key not in kpis: continue
        own, away, is_pct = kpis[key]

        left_txt  = f"★  {own:.1f}%" if is_pct else f"★  {int(own)}"
        right_txt = f"{away:.1f}%  ★" if is_pct else f"{int(away)}  ★"

        y = y0 - i*y_step
        ax.text(0.07, y, left_txt,  color="white", fontsize=9, ha="left",  va="center")
        ax.text(0.93, y, right_txt, color="white", fontsize=9, ha="right", va="center")
        ax.text(0.5,  y+0.013, key, color="white", fontsize=10, ha="center", va="center")

        # ancho relativo
        if is_pct:
            own_norm  = min(max(own, 0), 100) / 100.0
            away_norm = min(max(away,0), 100) / 100.0
        else:
            m = max(own, away, 1)
            own_norm  = own  / m
            away_norm = away / m

        # barra izquierda (propia) y derecha (rival)
        # centro 0.5; mitad izq 0.36..0.5 ; mitad der 0.5..0.64
        left_len  = 0.14 * own_norm
        right_len = 0.14 * away_norm
        ax.barh(y-0.012, left_len,  height=0.010, color=COLOR_HOME, left=0.5-left_len, align="edge")
        ax.barh(y-0.012, right_len, height=0.010, color=COLOR_AWAY, left=0.5,           align="edge")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================================================
# UI MENÚ (vamos a construir las otras páginas después)
# =========================================================
st.sidebar.header("Opciones")
menu = st.sidebar.radio(
    "Menú",
    ["📊 Estadísticas de Partido", "⏱️ Minutos", "🎯 Tiros", "🗺️ Mapa 3x3", "🔗 Red de Pases", "⚡ Radar"],
    index=0
)

# Banner si existe
banner = first_banner()
if banner is not None:
    st.image(banner, use_container_width=True)

st.title("InsightFutsal")

# =========================================================
# PÁGINA: ESTADÍSTICAS DE PARTIDO
# =========================================================
if menu == "📊 Estadísticas de Partido":
    files = list_xml_files(DATA_DIR)
    if not files:
        st.error(f"No hay XML con 'TotalValues' en {DATA_DIR}")
        st.stop()

    # Mapeo "bonito" -> filename real
    pretty = [pretty_match_name(f) for f in files]
    sel_pretty = st.selectbox("Elegí partido", pretty, index=0)
    fname = files[pretty.index(sel_pretty)]

    rival = extract_rival(fname)
    xml_path = os.path.join(DATA_DIR, fname)

    # Escudos
    ferro_badge = badge_for("Ferro")
    rival_badge = badge_for(rival)

    # Parseo + KPIs
    df = parse_asxml(xml_path)
    if df.empty:
        st.warning("No se pudo leer el XML o no tiene instancias.")
        st.stop()

    kpis = compute_kpis(df)

    # Panel KEY STATS
    draw_key_stats_panel(TITLE_HOME, rival, kpis, ferro_badge, rival_badge)

    # Debug opcional
    with st.expander("Ver tabla base (debug)"):
        st.dataframe(df.head(300), use_container_width=True)

# =========================================================
# PLACEHOLDERS (se implementan después)
# =========================================================
elif menu == "⏱️ Minutos":
    st.info("Esta sección se adaptará del notebook de Minutos por Jugador y Rol.")
elif menu == "🎯 Tiros":
    st.info("Esta sección se adaptará del notebook de Tiros por equipo/jugador.")
elif menu == "🗺️ Mapa 3x3":
    st.info("Esta sección se adaptará del notebook de Mapa de Calor.")
elif menu == "🔗 Red de Pases":
    st.info("Esta sección se adaptará del notebook de Red de Pases.")
elif menu == "⚡ Radar":
    st.info("Esta sección se adaptará del notebook de Radar por jugador.")
