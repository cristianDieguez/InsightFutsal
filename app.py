# app.py — InsightFutsal: Panel "KEY STATS" desde data/minutos (sin uploader)
import os, re, glob, math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from lxml import etree

# =========================
# CONFIG BÁSICA
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")

DATA_DIR = "data/minutos"
BADGE_DIR = "images/equipos"
BANNER_DIR = "images/banner"

# ===== AJUSTES DE ETIQUETAS EXACTAS =====
# Cambiá estas cadenas para calzar 1:1 con tu notebook/NacSport.
CODE_POSESION_FERRO = "Tiempo Posecion Ferro"
CODE_POSESION_RIVAL = "Tiempo Posecion Rival"

LABELS = {
    # nombre_metric: (lista_keywords_ferro, lista_keywords_rival, tipo, es_porcentaje)
    # tipo: "count" (conteo de instancias que contengan esos labels)
    #       "sum"   (suma de valores si existieran; por defecto no se usa)
    # es_porcentaje: True si debemos mostrar % y no valor absoluto
    "Posesión %": ([], [], "percent_possession", True),

    "Pases totales": (["pases totales", "pase total"], ["pases totales rival", "pase total rival"], "count", False),
    "Pases OK %": (["pase ok"], ["pase ok rival"], "ratio_pases_ok", True),
    "Pases último tercio": (["pase ultimo tercio"], ["pase ultimo tercio rival"], "count", False),
    "Pases al área": (["pase al area"], ["pase al area rival"], "count", False),

    "Tiros": (["tiro", "finalizacion jugada ctiro", "stiro"], ["tiro rival"], "count", False),
    "Tiros al arco": (["tiro al arco"], ["tiro al arco rival"], "count", False),

    "Recuperaciones": (["recuperacion"], ["recuperacion rival"], "count", False),
    "Duelos ganados": (["duelo ganado"], ["duelo ganado rival"], "count", False),
    "% Duelos ganados": (["duelo ganado"], ["duelo ganado rival"], "ratio_duelos", True),

    "Corners": (["corner"], ["corner rival"], "count", False),
    "Faltas": (["falta"], ["falta rival"], "count", False),
    "Goles": (["goles a favor en cancha", "gol a favor"], ["gol rival en cancha", "gol rival"], "count_unique_goal", False),
    "Asistencias": (["asistencia"], ["asistencia rival"], "count", False),
    "Pases clave": (["pase clave"], ["pase clave rival"], "count", False),
}

TITLE_HOME = "FERRO"   # cómo querés mostrar al equipo local (para el título grande)
PANEL_BG = "#0f5b33"   # verde panel
COLOR_HOME = "#ff8c2a" # barras propias (naranja similar a tu ejemplo)
COLOR_AWAY = "#96a1a1" # barras rival (gris)

# =========================
# UTILIDADES
# =========================
def list_xml_files(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(".xml") and "totalvalues" in f.lower()]
    files.sort()
    return files

def open_any(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None

def first_banner() -> Optional[Image.Image]:
    if not os.path.isdir(BANNER_DIR):
        return None
    for ext in ("png", "jpg", "jpeg", "webp"):
        for p in glob.glob(os.path.join(BANNER_DIR, f"*.{ext}")):
            img = open_any(p)
            if img: return img
    return None

def badge_for(team: str) -> Optional[Image.Image]:
    if not os.path.isdir(BADGE_DIR):
        return None
    name = team.strip().lower()
    # probá exacto y "slug"
    candidates = []
    for ext in ("png", "jpg", "jpeg", "webp"):
        candidates.append(os.path.join(BADGE_DIR, f"{name}.{ext}"))
        candidates.append(os.path.join(BADGE_DIR, f"{re.sub(r'\\s+', '_', name)}.{ext}"))
        candidates.append(os.path.join(BADGE_DIR, f"{re.sub(r'\\s+', '', name)}.{ext}"))
    for p in candidates:
        if os.path.isfile(p):
            img = open_any(p)
            if img: return img
    return None

def extract_rival(fname: str) -> str:
    rival = re.sub(r"(?i)^fecha\\s*\\d+\\s*-\\s*", "", fname)
    rival = re.sub(r"(?i)\\s*-\\s*xml\\s*totalvalues\\.xml$", "", rival)
    return rival.strip()

def labels_text_list(node) -> List[str]:
    return [t.text for t in node.findall(".//label/text") if t is not None and t.text]

def normtext(s: str) -> str:
    return (s or "").strip().lower()

# =========================
# PARSER asXML TotalValues → DataFrame
# =========================
@st.cache_data(show_spinner=False)
def parse_asxml(xml_path: str) -> pd.DataFrame:
    try:
        with open(xml_path, "rb") as fh:
            xml_bytes = fh.read()
        root = etree.fromstring(xml_bytes)
        inst = root.findall(".//ALL_INSTANCES/instance")
        if not inst: inst = root.findall(".//instance")

        rows = []
        for it in inst:
            start = it.findtext("start"); end = it.findtext("end")
            code  = normtext(it.findtext("code"))
            px = it.findtext("pos_x"); py = it.findtext("pos_y")
            labs = [normtext(x) for x in labels_text_list(it)]

            rows.append({
                "start": float(start) if start else np.nan,
                "end": float(end) if end else np.nan,
                "dur": (float(end)-float(start)) if start and end else np.nan,
                "code": code,
                "pos_x": float(px) if px else np.nan,
                "pos_y": float(py) if py else np.nan,
                "labels": labs
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error parseando {xml_path}: {e}")
        return pd.DataFrame(columns=["start","end","dur","code","pos_x","pos_y","labels"])

# =========================
# CÁLCULO DE KPI (ajustado con LABELS/CODES)
# =========================
def _has_any(labels: List[str], needles: List[str]) -> bool:
    if not needles: return False
    tt = " | ".join(labels)
    return any(n.lower() in tt for n in needles)

def compute_possession(df: pd.DataFrame) -> Tuple[float,float]:
    own = df.loc[df["code"].str.contains(normtext(CODE_POSESION_FERRO), na=False), "dur"].sum()
    away = df.loc[df["code"].str.contains(normtext(CODE_POSESION_RIVAL), na=False), "dur"].sum()
    total = own + away
    if total <= 0: return 0.0, 0.0
    p_own = round(own/total*100, 1); p_away = 100 - p_own
    return p_own, round(p_away,1)

def count_by_labels(df: pd.DataFrame, keywords: List[str]) -> int:
    if not keywords: return 0
    return int(df["labels"].apply(lambda L: _has_any(L, keywords)).sum())

def count_unique_goal(df: pd.DataFrame, keywords: List[str]) -> int:
    sub = df[df["labels"].apply(lambda L: _has_any(L, keywords))][["start","end"]].dropna()
    if sub.empty: return 0
    pairs = set(sub.itertuples(index=False, name=None))
    return len(pairs)

def ratio(a:int, b:int) -> float:
    return round((a/b*100.0),1) if b>0 else 0.0

def compute_kpis(df: pd.DataFrame) -> Dict[str, Tuple[float,float,bool]]:
    """
    Devuelve dict metric -> (valor_home, valor_away, es_porcentaje)
    """
    out = {}
    # Posesión
    own_pos, away_pos = compute_possession(df)
    out["Posesión %"] = (own_pos, away_pos, True)

    # Pases totales + Pases OK %
    pases_home = count_by_labels(df, LABELS["Pases totales"][0])
    pases_away = count_by_labels(df, LABELS["Pases totales"][1])
    out["Pases totales"] = (pases_home, pases_away, False)

    pases_ok_home = count_by_labels(df, LABELS["Pases OK %"][0])
    pases_ok_away = count_by_labels(df, LABELS["Pases OK %"][1])
    out["Pases OK %"] = (ratio(pases_ok_home, max(pases_home,1e-9)),
                         ratio(pases_ok_away, max(pases_away,1e-9)), True)

    # Resto genérico (conteos)
    for key in [
        "Pases último tercio","Pases al área","Tiros","Tiros al arco",
        "Recuperaciones","Duelos ganados","Corners","Faltas","Asistencias","Pases clave"
    ]:
        own = count_by_labels(df, LABELS[key][0])
        away = count_by_labels(df, LABELS[key][1])
        out[key] = (own, away, LABELS[key][3])

    # % Duelos ganados -> usa duelos ganados / (duelos ganados propios + rivales)
    own_duelos = out["Duelos ganados"][0]
    away_duelos = out["Duelos ganados"][1]
    total_duelos = own_duelos + away_duelos
    own_duelos_pct = round(own_duelos/total_duelos*100,1) if total_duelos>0 else 0.0
    away_duelos_pct = 100 - own_duelos_pct if total_duelos>0 else 0.0
    out["% Duelos ganados"] = (own_duelos_pct, away_duelos_pct, True)

    # Goles (únicos por start/end)
    own_goals = count_unique_goal(df, LABELS["Goles"][0])
    away_goals = count_unique_goal(df, LABELS["Goles"][1])
    out["Goles"] = (own_goals, away_goals, False)

    return out

# =========================
# RENDER DEL PANEL (estilo imagen de ejemplo)
# =========================
def draw_key_stats_panel(home_name: str, away_name: str, kpis: Dict[str, Tuple[float,float,bool]],
                         home_badge: Optional[Image.Image], away_badge: Optional[Image.Image]):
    # Orden de métricas (podés editar para coincidir exactamente con tu diseño)
    order = [
        "Posesión %","Pases totales","Pases OK %","Pases último tercio","Pases al área",
        "Tiros","Tiros al arco","Recuperaciones","Duelos ganados","% Duelos ganados",
        "Corners","Faltas","Goles","Asistencias","Pases clave"
    ]

    n = len(order)
    h = 1.2 + n*0.42 + 1.0
    fig = plt.figure(figsize=(10, h), dpi=200)
    ax = plt.gca()
    ax.set_facecolor(PANEL_BG)
    fig.patch.set_facecolor(PANEL_BG)
    ax.axis("off")

    # Título
    title = f"{home_name.upper()} vs {away_name.upper()}"
    ax.text(0.5, 1.02, title, ha="center", va="top", color="white",
            fontsize=22, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.98, "KEY STATS", ha="center", va="top", color="white",
            fontsize=13, transform=ax.transAxes)

    # Escudos
    if home_badge:
        ax.imshow(home_badge, extent=[0.02, 0.12, 0.93, 1.05], aspect="auto")
    if away_badge:
        ax.imshow(away_badge, extent=[0.88, 0.98, 0.93, 1.05], aspect="auto")

    # Barras
    y0 = 0.90
    y_step = 0.042
    bar_len = 0.28
    for i, key in enumerate(order):
        if key not in kpis: continue
        own, away, is_pct = kpis[key]

        # Etiquetas laterales
        left_txt  = f"★  {own:.1f}%" if is_pct else f"★  {int(own)}"
        right_txt = f"{away:.1f}%  ★" if is_pct else f"{int(away)}  ★"

        y = y0 - i*y_step
        ax.text(0.07, y, left_txt, color="white", fontsize=9, ha="left", va="center")
        ax.text(0.93, y, right_txt, color="white", fontsize=9, ha="right", va="center")

        # Barra central + nombre métrica
        ax.text(0.5, y+0.013, key, color="white", fontsize=10, ha="center", va="center")

        # escala 0..100 si es porcentaje; si no, normalizamos con el máximo entre ambos
        if is_pct:
            own_norm = min(own, 100)/100.0
            away_norm = min(away,100)/100.0
        else:
            m = max(own, away, 1)
            own_norm = own/m
            away_norm = away/m

        # barras (propia a la izquierda, rival a la derecha)
        ax.barh(y-0.012, own_norm, height=0.010, color=COLOR_HOME, left=0.36-own_norm*bar_len, align="center")
        ax.barh(y-0.012, away_norm, height=0.010, color=COLOR_AWAY, left=
