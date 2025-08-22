# app.py — InsightFutsal (Optimizado, misma lógica/outputs)
import os, re, glob, math, unicodedata
from typing import Optional, Tuple, Dict, List
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle as MplCircle
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image
import seaborn as sns

# =========================
# CONFIG UI / RUTAS
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")
st.title("InsightFutsal")

DATA_MINUTOS = "data/minutos"   # XML NacSport / XML TotalValues
DATA_MATRIX  = "data/matrix"    # Matrix.xlsx / Matrix.csv
BADGE_DIR    = "images/equipos" # logos
BANNER_DIR   = "images/banner"  # opcional

# Estilo panel
bg_green = "#006633"
text_w = "#FFFFFF"
bar_white = "#FFFFFF"
bar_rival = "#E6EEF2"
bar_rail = "#0F5E29"
star_c = "#FFD54A"
loser_alpha = 0.35
orange_win = "#FF8F00"
USE_ORANGE_FOR_WIN = True
RAISE_LABELS = True
BAR_HEIGHT_FACTOR = 0.36
LABEL_Y_SHIFT_LOW = 0.60
LABEL_Y_SHIFT_HIGH = 0.37
TRIM_LOGO_BORDERS = True
BANNER_H = 0.145
LOGO_W = 0.118
TITLE_FS = 32
SUB_FS = 19
FOOTER_H = 0.120
FOOTER_LOGO_W = 0.110
FOOTER_TITLE_FS = 20
FOOTER_SUB_FS = 14

mpl.rcParams.update({
    "savefig.facecolor": bg_green,
    "figure.facecolor": bg_green,
    "axes.facecolor": bg_green,
    "text.color": text_w,
})

# Orden y tipos
ROW_ORDER = [
    "Posesión %",
    "Pases totales",
    "Pases OK %",
    "Pases último tercio",
    "Pases al área",
    "Tiros",
    "Tiros al arco",
    "Recuperaciones",
    "Duelos ganados",
    "% Duelos ganados",
    "Corners",
    "Faltas",
    "Goles",
    "Asistencias",
    "Pases clave",
]
PERCENT_ROWS = {"Posesión %", "Pases OK %", "% Duelos ganados"}

# =========================
# HELPERS GENERALES
# =========================
def ntext(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.strip()

def nlower(s): return ntext(s).lower()

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

_NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")
def split_name_role(code: str) -> Tuple[Optional[str], Optional[str]]:
    if not code: return (None, None)
    m = _NAME_ROLE_RE.match(str(code))
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)

# Códigos de equipo rival / tiempos a excluir en distintos módulos
_EXCLUDE_PREFIXES = tuple([
    "categoria - equipo rival", "categoría - equipo rival",
    "tiempo posecion ferro", "tiempo posesion ferro",
    "tiempo posecion rival", "tiempo posesion rival",
    "tiempo no jugado",
])

def is_player_code(code: str) -> bool:
    if not code: return False
    code_norm = strip_accents(code).lower().strip()
    if any(code_norm.startswith(pref) for pref in _EXCLUDE_PREFIXES): return False
    return _NAME_ROLE_RE.match(code) is not None

def badge_path_for(name: str) -> Optional[str]:
    nm = name.strip().lower()
    cands = [
        os.path.join(BADGE_DIR, f"{nm}.png"),
        os.path.join(BADGE_DIR, f"{nm}.jpg"),
        os.path.join(BADGE_DIR, f"{nm.replace(' ','_')}.png"),
        os.path.join(BADGE_DIR, f"{nm.replace(' ','')}.png"),
        os.path.join(BADGE_DIR, f"{nm.replace(' ','-')}.png"),
    ]
    for p in cands:
        if os.path.isfile(p): return p
    return None

# =========================
# DISCOVER / MATCHES
# =========================
def list_matches() -> List[str]:
    if not os.path.isdir(DATA_MINUTOS): return []
    pats = glob.glob(os.path.join(DATA_MINUTOS, "Fecha * - * - XML TotalValues.xml"))
    labels = []
    rx = re.compile(r"^Fecha\s*([\d]+)\s*-\s*(.+?)\s*-\s*XML TotalValues\.xml$", re.I)
    for p in sorted(pats):
        base = os.path.basename(p); m = rx.match(base)
        if not m: continue
        fecha, rival = m.group(1).strip(), m.group(2).strip()
        labels.append(f"Fecha {fecha} - {rival}")
    return labels

def rival_from_label(label: str) -> str:
    parts = [p.strip() for p in label.split(" - ", 1)]
    return parts[1] if len(parts) == 2 else label

def infer_paths_for_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    xml_path = os.path.join(DATA_MINUTOS, f"{label} - XML TotalValues.xml")
    mx_xlsx = os.path.join(DATA_MATRIX, f"{label} - Matrix.xlsx")
    mx_csv  = os.path.join(DATA_MATRIX,  f"{label} - Matrix.csv")
    matrix_path = mx_xlsx if os.path.isfile(mx_xlsx) else (mx_csv if os.path.isfile(mx_csv) else None)
    return (xml_path if os.path.isfile(xml_path) else None), matrix_path

@st.cache_data(show_spinner=False)
def discover_matches() -> List[Dict]:
    if not os.path.isdir(DATA_MINUTOS): return []
    pats = []
    pats += glob.glob(os.path.join(DATA_MINUTOS, "* - XML NacSport.xml"))
    pats += glob.glob(os.path.join(DATA_MINUTOS, "* - XML TotalValues.xml"))
    files = sorted(set(pats))
    buckets: Dict[str, List[str]] = {}
    for p in files:
        base = os.path.basename(p)
        label = re.sub(r"(?i)\s*-\s*xml\s*(nacsport|totalvalues)\.xml$", "", base).strip()
        buckets.setdefault(label, []).append(p)

    def pref_key(path: str) -> int:
        b = os.path.basename(path).lower()
        if "xml nacsport" in b: return 0
        if "totalvalues" in b:  return 1
        return 2

    matches = []
    for label, paths in buckets.items():
        pick = sorted(paths, key=pref_key)[0]
        parts = [x.strip() for x in label.split(" - ")]
        rival = parts[-1] if parts else label
        mx_xlsx = os.path.join(DATA_MATRIX, f"{label} - Matrix.xlsx")
        mx_csv  = os.path.join(DATA_MATRIX,  f"{label} - Matrix.csv")
        matrix_path = mx_xlsx if os.path.isfile(mx_xlsx) else (mx_csv if os.path.isfile(mx_csv) else None)
        matches.append({
            "label": label,
            "xml_players": pick,  # NacSport preferente; si no, TotalValues
            "rival": rival,
            "matrix_path": matrix_path,
            "xml_equipo": None,
        })

    def date_key(m):
        mlabel = m["label"].lower()
        mnum = re.search(r"fecha\s*([0-9]+)", mlabel)
        return (0, int(mnum.group(1))) if mnum else (1, m["label"])

    return sorted(matches, key=date_key)

def get_match_by_label(label: str) -> Optional[Dict]:
    for m in discover_matches():
        if m["label"] == label: return m
    return None

# =========================
# XML LOADERS & COMMON PARSERS
# =========================
def parse_instances_generic(xml_path: str) -> List[dict]:
    """Devuelve instancias genéricas: code, labels_lc, start, end, xs, ys."""
    if not xml_path or not os.path.isfile(xml_path): return []
    root = ET.parse(xml_path).getroot()
    out = []
    for inst in root.findall(".//instance"):
        code = ntext(inst.findtext("code"))
        try:  st = float(inst.findtext("start") or "0")
        except: st = 0.0
        try:  en = float(inst.findtext("end") or "0")
        except: en = st
        labels_lc = [nlower(t.text) for t in inst.findall("./label/text")]
        xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
        ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
        out.append({"code": code, "labels_lc": labels_lc, "start": st, "end": en, "xs": xs, "ys": ys})
    return out

def parse_possession_from_equipo(xml_path: str) -> Tuple[float, float]:
    """Posesión desde XML (duración de instancias code==tiempo posecion ferro/rival)."""
    if not xml_path or not os.path.isfile(xml_path): return 0.0, 0.0
    root = ET.parse(xml_path).getroot()
    t_ferro = t_rival = 0.0
    for inst in root.findall(".//instance"):
        code = nlower(inst.findtext("code"))
        try:
            st = float(inst.findtext("start") or "0")
            en = float(inst.findtext("end") or "0")
        except:
            continue
        dur = max(0.0, en - st)
        if code == "tiempo posecion ferro": t_ferro += dur
        elif code == "tiempo posecion rival": t_rival += dur
    tot = t_ferro + t_rival
    if tot <= 0: return 0.0, 0.0
    return round(100*t_ferro/tot,1), round(100*t_rival/tot,1)

def load_matrix(path: str) -> Tuple[pd.DataFrame, str, Dict[str,str]]:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path, header=0)
    elif path.lower().endswith(".csv"):
        try: df = pd.read_csv(path, header=0)
        except Exception: df = pd.read_csv(path, header=0, sep=";")
    else:
        raise ValueError("Formato no soportado para MATRIX (usa .xlsx o .csv)")
    player_col = df.columns[0]
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    header_map = {c: nlower(c) for c in df.columns}
    return df, player_col, header_map

def sum_cols(df, who_mask, header_map, col_regex):
    cols = [c for c in df.columns[1:] if re.search(col_regex, header_map[c], re.I)]
    if not cols: return 0
    return int(df.loc[who_mask, cols].sum(numeric_only=True).sum())

def compute_from_matrix(path: str) -> Dict[str, Dict[str, float]]:
    df, player_col, header_map = load_matrix(path)
    who_raw = df[player_col]
    is_valid = who_raw.notna() & (who_raw.astype(str).str.strip() != "")
    who = who_raw.fillna("").astype(str).map(nlower)
    is_rival = is_valid & (who == "categoria - equipo rival")
    is_mine  = is_valid & ~is_rival

    pases_tot_re = r"^pase.*frontal|^pase.*lateral"
    pases_ok_re  = r"^pase.*completado"
    tiros_re     = r"\btiro hecho\b"
    taro_re      = r"\btiro atajado\b"
    recup_re     = r"^recuperacion\b"
    duelo_g_re   = r"\brecuperacion x duelo\b"
    duelo_p_re   = r"\bperdida x duelo\b"
    corners_re   = r"\bcorner en corto\b|\bcorner al area\b|\bcorner 2do palo\b"
    faltas_re    = r"\bfaltas recibidas\b"
    goles_re     = r"^gol\b"
    asis_re      = r"\basistencia\b"
    pclave_re    = r"\bpase clave\b"

    def pct(a,b): return round(100*a/b,1) if b else 0.0

    # FERRO
    pases_tot_m = sum_cols(df, is_mine,  header_map, pases_tot_re)
    pases_ok_m  = sum_cols(df, is_mine,  header_map, pases_ok_re)
    tiros_m     = sum_cols(df, is_mine,  header_map, tiros_re)
    taro_m      = sum_cols(df, is_mine,  header_map, taro_re)
    recup_m     = sum_cols(df, is_mine,  header_map, recup_re)
    duelo_g_m   = sum_cols(df, is_mine,  header_map, duelo_g_re)
    duelo_p_m   = sum_cols(df, is_mine,  header_map, duelo_p_re)
    corners_m   = sum_cols(df, is_mine,  header_map, corners_re)
    faltas_m    = sum_cols(df, is_mine,  header_map, faltas_re)
    goles_m     = sum_cols(df, is_mine,  header_map, goles_re)
    asis_m      = sum_cols(df, is_mine,  header_map, asis_re)
    pclave_m    = sum_cols(df, is_mine,  header_map, pclave_re)

    # RIVAL
    pases_tot_r = sum_cols(df, is_rival, header_map, pases_tot_re)
    pases_ok_r  = sum_cols(df, is_rival, header_map, pases_ok_re)
    tiros_r     = sum_cols(df, is_rival, header_map, tiros_re)
    taro_r      = sum_cols(df, is_rival, header_map, taro_re)
    recup_r     = sum_cols(df, is_rival, header_map, recup_re)
    duelo_g_r   = sum_cols(df, is_rival, header_map, duelo_g_re)
    duelo_p_r   = sum_cols(df, is_rival, header_map, duelo_p_re)
    corners_r   = sum_cols(df, is_rival, header_map, corners_re)
    faltas_r    = sum_cols(df, is_rival, header_map, faltas_re)
    goles_r     = sum_cols(df, is_rival, header_map, goles_re)
    asis_r      = sum_cols(df, is_rival, header_map, asis_re)
    pclave_r    = sum_cols(df, is_rival, header_map, pclave_re)

    return {
        "FERRO": {
            "Pases totales": pases_tot_m,
            "Pases OK %": pct(pases_ok_m, pases_tot_m),
            "Tiros": tiros_m,
            "Tiros al arco": taro_m,
            "Recuperaciones": recup_m,
            "Duelos Ganados": duelo_g_m,
            "% Duelos Ganados": pct(duelo_g_m, duelo_g_m + duelo_p_m),
            "Corners": corners_m,
            "Faltas": faltas_m,
            "Goles": goles_m,
            "Asistencias": asis_m,
            "Pases Clave": pclave_m,
        },
        "RIVAL": {
            "Pases totales": pases_tot_r,
            "Pases OK %": pct(pases_ok_r, pases_tot_r),
            "Tiros": tiros_r,
            "Tiros al arco": taro_r,
            "Recuperaciones": recup_r,
            "Duelos Ganados": duelo_g_r,
            "% Duelos Ganados": pct(duelo_g_r, duelo_g_r + duelo_p_r),
            "Corners": corners_r,
            "Faltas": faltas_r,
            "Goles": goles_r,
            "Asistencias": asis_r,
            "Pases Clave": pclave_r,
        }
    }

# =========================
# CANCHA / LOGOS / PLOT BASE
# =========================
def load_any_image(path):
    im = Image.open(path); im.load()
    if im.mode != "RGBA": im = im.convert("RGBA")
    return np.array(im)

def trim_margins(img_rgba, bg_tol=12):
    h, w, _ = img_rgba.shape
    alpha = img_rgba[:,:,3]
    rgb = img_rgba[:,:,:3].astype(np.int16)
    white_mask = (np.abs(255 - rgb).max(axis=2) <= bg_tol)
    useful = (~white_mask) | (alpha > 0)
    rows = np.where(useful.any(axis=1))[0]
    cols = np.where(useful.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0: return img_rgba
    r0, r1 = rows[0], rows[-1]; c0, c1 = cols[0], cols[-1]
    return img_rgba[r0:r1+1, c0:c1+1, :]

def draw_logo(ax, path, cx, cy, width):
    try:
        img = load_any_image(path)
        if TRIM_LOGO_BORDERS: img = trim_margins(img)
    except Exception:
        return
    h, w = img.shape[0], img.shape[1]
    aspect = h / w if w else 1.0
    w_norm, h_norm = width, width * aspect
    ax.imshow(img, extent=[cx - w_norm/2, cx + w_norm/2, cy - h_norm/2, cy + h_norm/2], zorder=6)

# Cancha futsal horizontal (35 x 20)
ANCHO, ALTO = 35.0, 20.0
def draw_pitch(ax, grid=False, with_marks=True):
    ax.set_facecolor("white")
    ax.plot([0, ANCHO], [0, 0], color="black")
    ax.plot([0, ANCHO], [ALTO, ALTO], color="black")
    ax.plot([0, 0], [0, ALTO], color="black")
    ax.plot([ANCHO, ANCHO], [0, ALTO], color="black")
    ax.plot([ANCHO/2, ANCHO/2], [0, ALTO], color="black")
    if with_marks:
        ax.add_patch(Arc((0, ALTO/2), 8, 12, angle=0, theta1=270, theta2=90, color="black"))
        ax.plot([0,1.0],[8.5,8.5], color="black", linewidth=2)
        ax.plot([0,1.0],[11.5,11.5], color="black", linewidth=2)
        ax.plot([1.0,1.0],[8.5,11.5], color="black", linewidth=2)
        ax.add_patch(Arc((ANCHO, ALTO/2), 8, 12, angle=0, theta1=90, theta2=270, color="black"))
        ax.plot([ANCHO, ANCHO-1.0],[8.5,8.5], color="black", linewidth=2)
        ax.plot([ANCHO, ANCHO-1.0],[11.5,11.5], color="black", linewidth=2)
        ax.plot([ANCHO-1.0, ANCHO-1.0],[8.5,11.5], color="black", linewidth=2)
        ax.add_patch(MplCircle((ANCHO/2, ALTO/2), 4, color="black", fill=False))
        ax.add_patch(MplCircle((ANCHO/2, ALTO/2), 0.2, color="black", fill=True))
    if grid:
        for j in range(3):
            for i in range(3):
                dx, dy = ANCHO/3, ALTO/3
                x0, y0 = i*dx, j*dy
                ax.add_patch(Rectangle((x0,y0), dx,dy, linewidth=0.6, edgecolor='gray', facecolor='none'))
                zona = j*3 + i + 1
                ax.text(x0 + dx - 0.4, y0 + dy - 0.4, str(zona), ha='right', va='top', fontsize=9, color='gray')
    ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis("off")

# =========================
# TIMELINE
# =========================
BALL_ICON_PATH    = os.path.join(BANNER_DIR, "pelota.png")
FOOTER_LEFT_LOGO  = os.path.join(BANNER_DIR, "SportData.png")
FOOTER_RIGHT_LOGO = os.path.join(BANNER_DIR, "Sevilla.png")
INCLUDE_GOALS_IN_SHOT_CIRCLES = True

def try_load_ball():
    try:
        im = Image.open(BALL_ICON_PATH); im.load()
        if im.mode != "RGBA": im = im.convert("RGBA")
        return np.array(im)
    except Exception:
        return None

BALL_IMG = try_load_ball()

def draw_ball(ax, x_center, y_center, size=0.018):
    if BALL_IMG is None:
        ax.text(x_center, y_center, "⚽", ha="center", va="center", fontsize=8)
        return
    h, wpx = BALL_IMG.shape[0], BALL_IMG.shape[1]
    asp = h / wpx if wpx else 1.0
    ax.imshow(
        BALL_IMG,
        extent=[x_center - size/2, x_center + size/2, y_center - (size*asp)/2, y_center + (size*asp)/2],
        zorder=9
    )

def is_pass_attempt(ev) -> bool:
    if re.match(r"^\s*pase\b", nlower(ev["code"])): return True
    return any(re.match(r"^\s*pase\b", l or "") for l in ev["labels_lc"])

def is_rival_code(code) -> bool:
    return nlower(code).startswith("categoria - equipo rival") or nlower(code).startswith("categoría - equipo rival")

def xy_to_zone(x, y, max_x=19, max_y=34):
    if x is None or y is None: return None
    col = 1 if x <= 6 else (2 if x <= 13 else 3)
    row = 1 if y <= 11 else (2 if y <= 22 else 3)
    return (row-1)*3 + col

ON_TARGET_PAT = re.compile(r"\b(al\s*arco|a\s*puerta|a\s*porter[ií]a|on\s*target|atajad[oa]|saved\s*shot)\b", re.I)
def is_shot(ev):
    s = nlower(ev["code"])
    if re.search(r"\btiro\b|\bremate\b", s): return True
    return any(re.search(r"\btiro\b|\bremate\b", l or "") for l in ev["labels_lc"])

def is_goal(ev):
    s = nlower(ev["code"])
    if re.match(r"^gol\b", s): return True
    return any(re.match(r"^gol\b", l or "") for l in ev["labels_lc"])

def is_shot_on_target(ev):
    s = nlower(ev["code"])
    if ON_TARGET_PAT.search(s): return True
    return any(ON_TARGET_PAT.search(l or "") for l in ev["labels_lc"])

def minute_bucket(sec):
    m = int(sec // 60)
    return max(0, min(39, m))  # 0..39

def parse_instances_jugadores(xml_path: str):
    # Reutiliza parse_instances_generic para compatibilidad previa
    evs = parse_instances_generic(xml_path)
    out = []
    for ev in evs:
        xs, ys = ev["xs"], ev["ys"]
        x_end, y_end = (xs[-1], ys[-1]) if xs and ys else (None, None)
        out.append({**ev, "end_xy": (x_end, y_end)})
    return out

def build_timeline(xml_players_path):
    evs = parse_instances_jugadores(xml_players_path)
    M = 40
    tl = dict(
        passes_M=np.zeros(M, int), passes_R=np.zeros(M, int),
        last_M=np.zeros(M, int),   last_R=np.zeros(M, int),
        shots_on_M=np.zeros(M, int), shots_off_M=np.zeros(M, int),
        shots_on_R=np.zeros(M, int), shots_off_R=np.zeros(M, int),
        goals_M=np.zeros(M, int), goals_R=np.zeros(M, int),
    )
    for ev in evs:
        m = minute_bucket(ev.get("end", ev.get("start", 0.0)))
        if is_pass_attempt(ev):
            if is_rival_code(ev["code"]): tl["passes_R"][m] += 1
            else:                          tl["passes_M"][m] += 1
            z = xy_to_zone(*(ev.get("end_xy") or (None, None)))
            if z is not None:
                if is_rival_code(ev["code"]):
                    if z in {1,4,7}: tl["last_R"][m] += 1
                else:
                    if z in {3,6,9}: tl["last_M"][m] += 1
        if is_shot(ev):
            goal = is_goal(ev)
            on_t = is_shot_on_target(ev) or goal
            if is_rival_code(ev["code"]):
                if goal:
                    tl["goals_R"][m] += 1
                    if INCLUDE_GOALS_IN_SHOT_CIRCLES: tl["shots_on_R"][m] += 1
                else:
                    (tl["shots_on_R"] if on_t else tl["shots_off_R"])[m] += 1
            else:
                if goal:
                    tl["goals_M"][m] += 1
                    if INCLUDE_GOALS_IN_SHOT_CIRCLES: tl["shots_on_M"][m] += 1
                else:
                    (tl["shots_on_M"] if on_t else tl["shots_off_M"])[m] += 1
    return tl

# Colores específicos del timeline
yellow_on = "#FFD54F"; rival_g = "#B9C4C9"; white = "#FFFFFF"; rail = "#0F5E29"

def draw_count_circle(ax, x, y, count, base_r=0.006, face=None, edge=white, lw=1.0, z=8):
    if count <= 0: return
    r = base_r * (count ** 0.25)  # mismo “look” que antes
    circ = MplCircle((x, y), radius=r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(circ)

def draw_timeline_panel(rival_name: str, tl: dict, ferro_logo_path: Optional[str], rival_logo_path: Optional[str]):
    plt.close("all")
    fig_h = 11.8
    fig = plt.figure(figsize=(10.8, fig_h))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0), 1, 1, facecolor=bg_green, edgecolor="none"))

    # Banner sup
    BANNER_H_loc = 0.14; BANNER_Y0 = 1 - BANNER_H_loc
    ax.add_patch(Rectangle((0, BANNER_Y0), 1, BANNER_H_loc, facecolor=white, edgecolor="none"))
    if ferro_logo_path: draw_logo(ax, ferro_logo_path, 0.075, BANNER_Y0+0.07, 0.12)
    if rival_logo_path: draw_logo(ax, rival_logo_path, 0.925, BANNER_Y0+0.07, 0.12)
    ax.text(0.5, BANNER_Y0+0.085, f"FERRO vs {rival_name.upper()}", ha="center", va="center", color=bg_green, fontsize=30, weight="bold")
    ax.text(0.5, BANNER_Y0+0.040, "TIMELINE", ha="center", va="center", color=bg_green, fontsize=16, weight="bold")

    # Banner inferior
    FOOT_H, FOOT_Y0 = 0.12, 0.00
    ax.add_patch(Rectangle((0, FOOT_Y0), 1, FOOT_H, facecolor=white, edgecolor="none"))
    if os.path.isfile(FOOTER_LEFT_LOGO):  draw_logo(ax, FOOTER_LEFT_LOGO, 0.09, FOOT_Y0+FOOT_H*0.52, 0.14)
    if os.path.isfile(FOOTER_RIGHT_LOGO): draw_logo(ax, FOOTER_RIGHT_LOGO, 0.91, FOOT_Y0+FOOT_H*0.52, 0.12)
    ax.text(0.50, FOOT_Y0+FOOT_H*0.62, "TRABAJO FIN DE MÁSTER", ha="center", va="center", color=bg_green, fontsize=18, weight="bold")
    ax.text(0.50, FOOT_Y0+FOOT_H*0.32, "Cristian Dieguez", ha="center", va="center", color=bg_green, fontsize=13, weight="bold")

    # Panel central
    panel_y0, panel_y1 = FOOT_Y0+FOOT_H+0.05, BANNER_Y0-0.05
    panel_h = panel_y1 - panel_y0
    x_center_gap_L = 0.47; x_center_gap_R = 0.53
    x_bar_M_max = 0.22;  x_bar_R_max  = 0.78
    x_shot_M = 0.05;     x_last_M = 0.16
    x_shot_R = 0.95;     x_last_R = 0.84
    x_goal_M = x_shot_M - 0.025
    x_goal_R = x_shot_R + 0.025

    # Títulos
    ty = panel_y1 + 0.012
    ax.text(x_last_M, ty, "Últ. tercio", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_bar_M_max+x_center_gap_L)/2, ty, "Pases/min", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_shot_M, ty, "Tiros / Goles", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_center_gap_L+x_center_gap_R)/2, ty, "Min.", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_center_gap_R+x_bar_R_max)/2, ty, "Pases/min", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_shot_R, ty, "Tiros / Goles", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_last_R, ty, "Últ. tercio", ha="center", va="bottom", fontsize=10, weight="bold")

    M = 40
    def y_of_index(i): return panel_y1 - panel_h * (i + 0.5) / M
    ax.add_line(plt.Line2D([x_center_gap_L, x_center_gap_L],[panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    ax.add_line(plt.Line2D([x_center_gap_R, x_center_gap_R],[panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    for m in range(0, 41, 5):
        yy = y_of_index(min(m, 39))
        ax.text(0.50, yy, f"{m:02d}'", ha="center", va="center", fontsize=8, alpha=0.90)

    max_bar = max(tl["passes_M"].max() if tl["passes_M"].size else 1,
                  tl["passes_R"].max() if tl["passes_R"].size else 1, 1)

    def bar_width_left(cnt):
        total_span = (x_center_gap_L - x_bar_M_max)
        return total_span * (cnt / max_bar if max_bar else 0)

    def bar_width_right(cnt):
        total_span = (x_bar_R_max - x_center_gap_R)
        return total_span * (cnt / max_bar if max_bar else 0)

    bar_h = panel_h / M * 0.55
    for m in range(M):
        y = y_of_index(m)
        # FERRO izq
        wL = bar_width_left(tl["passes_M"][m]); x0L = x_center_gap_L - wL
        ax.add_patch(Rectangle((x0L, y - bar_h/2), wL, bar_h, facecolor=orange_win, edgecolor="none"))
        if tl["passes_M"][m] > 0:
            ax.text(x0L - 0.006, y, f"{tl['passes_M'][m]}", ha="right", va="center", fontsize=8)
        draw_count_circle(ax, x_last_M, y, tl["last_M"][m], base_r=0.006, face=white, edge=white, lw=0.0, z=7)
        draw_count_circle(ax, x_shot_M, y, tl["shots_off_M"][m], base_r=0.006, face=None, edge=white, lw=1.0, z=8)
        draw_count_circle(ax, x_shot_M, y, tl["shots_on_M"][m],  base_r=0.006, face=yellow_on, edge=white, lw=0.6, z=9)
        if tl["goals_M"][m] > 0:
            for k in range(int(tl["goals_M"][m])):
                draw_ball(ax, x_goal_M - k*0.012, y, size=0.016)

        # RIVAL der
        wR = bar_width_right(tl["passes_R"][m]); x0R = x_center_gap_R
        ax.add_patch(Rectangle((x0R, y - bar_h/2), wR, bar_h, facecolor=rival_g, edgecolor="none"))
        if tl["passes_R"][m] > 0:
            ax.text(x0R + wR + 0.006, y, f"{tl['passes_R'][m]}", ha="left", va="center", fontsize=8)
        draw_count_circle(ax, x_last_R, y, tl["last_R"][m], base_r=0.006, face=white, edge=white, lw=0.0, z=7)
        draw_count_circle(ax, x_shot_R, y, tl["shots_off_R"][m], base_r=0.006, face=None, edge=white, lw=1.0, z=8)
        draw_count_circle(ax, x_shot_R, y, tl["shots_on_R"][m],  base_r=0.006, face=yellow_on, edge=white, lw=0.6, z=9)
        if tl["goals_R"][m] > 0:
            for k in range(int(tl["goals_R"][m])):
                draw_ball(ax, x_goal_R + k*0.012, y, size=0.016)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================
# MAPAS DE CALOR
# =========================
# Keywords de pases (mismo criterio)
KEYWORDS_PASES = [k.lower() for k in [
    "pase corto frontal","pase corto lateral","pase largo frontal","pase largo lateral",
    "pase progresivo frontal","pase progresivo lateral",
    "pase progresivo frontal cpie","pase progresivo lateral cpie",
    "salida de arco progresivo cmano","pase corto frontal cpie",
    "pase corto lateral cpie","salida de arco corto cmano"
]]

def cargar_datos_xml_coordenadas(xml_path: str) -> pd.DataFrame:
    """code (jugador), pos_x_list, pos_y_list, labels(list) desde XML."""
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["jugador","pos_x_list","pos_y_list","labels"])
    root = ET.parse(xml_path).getroot()
    data = []
    for inst in root.findall(".//instance"):
        jugador = ntext(inst.findtext("code"))
        pos_x = [float(px.text) for px in inst.findall("pos_x") if (px.text or "").strip()!=""]
        pos_y = [float(py.text) for py in inst.findall("pos_y") if (py.text or "").strip()!=""]
        labels = [ntext(lbl.findtext("text")) for lbl in inst.findall("label")]
        data.append({"jugador": jugador, "pos_x_list": pos_x, "pos_y_list": pos_y, "labels": labels})
    return pd.DataFrame(data)

def _es_evento_pase_o_tiro(labels: List[str]) -> bool:
    lbls = [nlower(l or "") for l in labels]
    is_shot = any("tiro" in l or "remate" in l for l in lbls)
    is_pass = any(any(k in l for k in KEYWORDS_PASES) for l in lbls)
    return is_shot or is_pass

def filtrar_coords_por_evento(row: pd.Series) -> Tuple[List[float], List[float]]:
    # Si es pase (keywords) o tiro -> solo primer punto; si no, todos.
    if _es_evento_pase_o_tiro(row.get("labels", [])):
        return row["pos_x_list"][:1], row["pos_y_list"][:1]
    return row["pos_x_list"], row["pos_y_list"]

def parse_player_and_role(code: str) -> Tuple[str, Optional[str]]:
    m = _NAME_ROLE_RE.match(code.strip()) if code else None
    if m:
        return ntext(m.group(1)).strip(), ntext(m.group(2)).strip()
    return code.strip(), None

def explode_coords_for_heatmap(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=["player_name","role","pos_x","pos_y"])
    tmp = df_raw.copy()
    tmp["pos_x_list"], tmp["pos_y_list"] = zip(*tmp.apply(filtrar_coords_por_evento, axis=1))
    tmp = tmp.explode(["pos_x_list","pos_y_list"], ignore_index=True)
    tmp = tmp.rename(columns={"pos_x_list":"pos_x","pos_y_list":"pos_y"})
    tmp = tmp.dropna(subset=["pos_x","pos_y"])
    names_roles = tmp["jugador"].apply(parse_player_and_role)
    tmp["player_name"] = names_roles.apply(lambda t: t[0])
    tmp["role"] = names_roles.apply(lambda t: t[1])
    tmp["player_name"] = tmp["player_name"].map(ntext)
    tmp["role"] = tmp["role"].map(lambda x: ntext(x) if x else None)
    return tmp[["player_name","role","pos_x","pos_y"]]

def rotate_coords_for_attack_right(df: pd.DataFrame, role: Optional[str]=None) -> pd.DataFrame:
    if df.empty: return df.copy()
    ancho_original, alto_original = 20, 35
    out = df.copy()
    out["x_rot"] = alto_original - out["pos_y"]
    out["y_rot"] = out["pos_x"]
    if role and role in {"Ala I","Ala D"}:
        out["y_rot"] = ancho_original - out["y_rot"]
    return out

def fig_heatmap(df_xy: pd.DataFrame, titulo: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax, grid=False, with_marks=True)
    if df_xy.empty:
        ax.text(0.5, 0.5, "Sin datos para el filtro", ha="center", va="center", transform=ax.transAxes)
        return fig
    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_heatmap", ["#a8dadc", "#bde0c6", "#ffe5a1", "#f6a192"]
    )
    sns.kdeplot(
        x=df_xy["x_rot"], y=df_xy["y_rot"], fill=True, cmap=pastel_cmap,
        bw_adjust=0.4, levels=50, alpha=0.75, ax=ax, clip=((0, 35), (0, 20))
    )
    ax.set_title(titulo, fontsize=14)
    return fig

# =========================
# MINUTOS — helpers
# =========================
def _merge_intervals(intervals):
    ints = [(float(s), float(e)) for s, e in intervals if s is not None and e is not None and e > s]
    if not ints: return []
    ints.sort()
    merged = [list(ints[0])]
    for s, e in ints[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    return [(s, e) for s, e in merged]

def cargar_minutos_desde_xml_totalvalues(xml_path: str) -> pd.DataFrame:
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["code","nombre","rol","start_s","end_s","dur_s"])
    root = ET.parse(xml_path).getroot()
    rows = []
    for inst in root.findall(".//instance"):
        code = inst.findtext("code") or ""
        if not is_player_code(code): continue
        m = _NAME_ROLE_RE.match(code); nombre, rol = m.group(1).strip(), m.group(2).strip()
        st = inst.findtext("start"); en = inst.findtext("end")
        try:
            s = float(st) if st is not None else None
            e = float(en) if en is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s: continue
        rows.append({"code": code, "nombre": nombre, "rol": rol, "start_s": s, "end_s": e, "dur_s": e - s})
    return pd.DataFrame(rows)

def _format_mmss(seconds: float | int) -> str:
    if seconds is None or not np.isfinite(seconds): return "00:00"
    s = int(round(float(seconds)))
    mm, ss = divmod(s, 60)
    return f"{mm:02d}:{ss:02d}"

def minutos_por_presencia(df_pres: pd.DataFrame):
    if df_pres.empty:
        return (pd.DataFrame(columns=["nombre","rol","segundos","mmss","minutos","n_tramos"]),
                pd.DataFrame(columns=["nombre","segundos","mmss","minutos","n_tramos"]))
    out = []
    for (nombre, rol), g in df_pres.groupby(["nombre","rol"], dropna=False):
        intervals = list(zip(g["start_s"], g["end_s"]))
        merged = _merge_intervals(intervals)
        secs = sum(e - s for s, e in merged)
        out.append({
            "nombre": nombre, "rol": rol, "segundos": int(round(secs)),
            "mmss": _format_mmss(secs), "minutos": secs/60.0, "n_tramos": len(merged)
        })
    df_por_rol = (pd.DataFrame(out)
                  .sort_values(["segundos","nombre"], ascending=[False, True])
                  .reset_index(drop=True))
    df_por_rol["minutos"] = df_por_rol["minutos"].round(2)
    df_por_jugador = (df_por_rol
        .groupby("nombre", as_index=False)
        .agg(segundos=("segundos","sum"), minutos=("minutos","sum"), n_tramos=("n_tramos","sum"))
        .assign(mmss=lambda d: d["segundos"].apply(_format_mmss))
        .sort_values(["segundos","nombre"], ascending=[False, True])
        .reset_index(drop=True))
    df_por_jugador["minutos"] = df_por_jugador["minutos"].round(2)
    return df_por_rol, df_por_jugador

def _inside_bar_label(ax, bars, values_sec):
    xlim = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else max(1, (np.array(values_sec)/60).max() if len(values_sec) else 1)
    for b, secs in zip(bars, values_sec):
        mmss = _format_mmss(secs)
        x = b.get_width(); y = b.get_y() + b.get_height()/2
        txt_x = x * 0.98 if x >= 0.9 else min(x + 0.1, xlim * 0.98)
        ha = "right" if x >= 0.9 else "left"
        color = "white" if x >= 0.9 else "black"
        ax.text(txt_x, y, mmss, va="center", ha=ha, fontsize=9, color=color, fontweight="bold")

def fig_barh_minutos(labels, vals_sec, title, xlabel="Minutos"):
    vals_min = (np.array(vals_sec) / 60.0)
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.48*len(labels))))
    bars = ax.barh(labels, vals_min, alpha=0.9)
    ax.invert_yaxis()
    _inside_bar_label(ax, bars, vals_sec)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    vmax = max(vals_min) if len(vals_min) else 1.0
    ax.set_xlim(0, vmax * 1.12 + 0.4)
    plt.tight_layout()
    return fig

# =========================
# RED DE PASES — helpers (optimizado, sin duplicar funciones)
# =========================
from collections import defaultdict

# NOTA: Reutiliza:
# - draw_futsal_pitch_horizontal(ax) ya definida antes
# - cargar_datos_nacsport(xml_path) ya definida antes (code, labels, pos_x_list, pos_y_list)

PASS_KEYWORDS = [
    "pase corto frontal", "pase corto lateral",
    "pase largo frontal", "pase largo lateral",
    "pase progresivo frontal", "pase progresivo lateral",
    "pase progresivo frontal cpie", "pase progresivo lateral cpie",
    "salida de arco progresivo cmano",
    "pase corto frontal cpie", "pase corto lateral cpie",
    "salida de arco corto cmano",
]
PASS_KEYWORDS = [k.lower() for k in PASS_KEYWORDS]

def red_de_pases_por_rol(df: pd.DataFrame) -> plt.Figure:
    """
    Construye la red de pases por ROL usando el primer punto de cada pase como origen.
    Reglas y rotaciones iguales al resto de vistas (35x20, ataque a la derecha).
    Mantiene la lógica original, sólo consolida y limpia duplicados.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_futsal_pitch_horizontal(ax)

    coords_origen = defaultdict(list)
    totales_hechos_por_rol = defaultdict(int)
    conteo_roles_total = defaultdict(int)

    for _, row in df.iterrows():
        code = row.get("jugador") or ""
        if not is_player_code(code):  # usa helper ya definido arriba
            continue

        labels = [(_strip_accents(lbl).lower() if lbl else "") for lbl in row.get("labels", [])]
        if not any(k in l for l in labels for k in PASS_KEYWORDS):
            continue

        m = _NAME_ROLE_RE.match(code)
        rol_origen = m.group(2).strip() if m else None
        if not rol_origen:
            continue

        # rol destino: primera etiqueta que incluya "(Rol)" distinta del origen
        rol_destino = None
        for lbl in row.get("labels", []):
            if lbl and "(" in lbl and ")" in lbl:
                pos = lbl.split("(")[1].replace(")", "").strip()
                if pos and pos != rol_origen:
                    rol_destino = pos
                    break

        px, py = row.get("pos_x_list") or [], row.get("pos_y_list") or []
        if px and py:
            # Rotación a cancha 35x20 (misma que otras vistas):
            # original: x∈[0..20], y∈[0..40] → cancha: X=35-Y/escala, Y= X/escala
            x0 = 35 - (py[0] * (35.0 / 40.0))
            y0 = px[0]
            coords_origen[rol_origen].append((x0, y0))
            totales_hechos_por_rol[rol_origen] += 1

        if rol_destino and rol_destino != rol_origen:
            key = tuple(sorted([rol_origen, rol_destino]))
            conteo_roles_total[key] += 1

    # promedios por rol (posición de nodo)
    rol_coords = {}
    for rol, coords in coords_origen.items():
        arr = np.array(coords)
        rol_coords[rol] = (arr[:, 0].mean(), arr[:, 1].mean())

    # Heurística original para ubicación de Arq y coherencia Ala I / Ala D
    if "Arq" in rol_coords:
        rol_coords["Arq"] = (3, 10)

    if "Ala I" in rol_coords and "Ala D" in rol_coords:
        yI, yD = rol_coords["Ala I"][1], rol_coords["Ala D"][1]
        if yI < yD:
            rol_coords["Ala I"], rol_coords["Ala D"] = rol_coords["Ala D"], rol_coords["Ala I"]
            # Ajustar los conteos e intercambiar totales manteniendo la lógica previa
            new_counts = defaultdict(int)
            for (a, b), v in conteo_roles_total.items():
                aa = "Ala D" if a == "Ala I" else ("Ala I" if a == "Ala D" else a)
                bb = "Ala D" if b == "Ala I" else ("Ala I" if b == "Ala D" else b)
                new_counts[tuple(sorted([aa, bb]))] += v
            conteo_roles_total = new_counts

            new_tot = defaultdict(int)
            for r, t in totales_hechos_por_rol.items():
                if r == "Ala I":
                    new_tot["Ala D"] = t
                elif r == "Ala D":
                    new_tot["Ala I"] = t
                else:
                    new_tot[r] = t
            totales_hechos_por_rol = new_tot

    # dibujar edges
    max_pases = max(conteo_roles_total.values()) if conteo_roles_total else 1
    for (ra, rb), count in conteo_roles_total.items():
        if ra in rol_coords and rb in rol_coords:
            x1, y1 = rol_coords[ra]; x2, y2 = rol_coords[rb]
            lw = 1 + (count / max_pases) * 5
            ax.plot([x1, x2], [y1, y2], linewidth=lw, alpha=0.7, zorder=3, color="#E74C3C")
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(count), fontsize=10,
                    ha='center', va='center', fontweight='bold', zorder=6, color="#154360")

    # nodos
    for rol, (x, y) in rol_coords.items():
        ax.scatter(x, y, s=2000, color='white', edgecolors='black', zorder=5)
        ax.text(x, y, f"{rol}\n{totales_hechos_por_rol.get(rol, 0)}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='black', zorder=7)

    ax.set_title("Red de Pases por Rol (NacSport/TotalValues; sólo Jugador (Rol))", fontsize=13)
    plt.tight_layout()
    return fig


# =========================
# UI — 🔗 Red de Pases
# =========================
elif menu == "🔗 Red de Pases":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    labels = [m["label"] for m in matches]
    sel = st.selectbox("Elegí partido", labels, index=0)
    m = get_match_by_label(sel)
    if not m:
        st.error("No pude resolver el partido seleccionado.")
        st.stop()

    if not m.get("xml_players") or not os.path.isfile(m["xml_players"]):
        st.warning("No encontré el XML NacSport/TotalValues del partido seleccionado.")
        st.stop()

    df_all = cargar_datos_nacsport(m["xml_players"])
    fig = red_de_pases_por_rol(df_all)
    st.pyplot(fig, use_container_width=True)


# =========================
# PÉRDIDAS & RECUPERACIONES — helpers (reutiliza y refactor mínimo)
# =========================
# NOTA: Todo el bloque PR_* ya estaba definido arriba en tu código largo.
# Aquí sólo confirmamos que se usa tal cual (sin cambiar lógica) y seguimos a la UI.

# =========================
# UI — 🛡️ Pérdidas y Recuperaciones
# =========================
elif menu == "🛡️ Pérdidas y Recuperaciones":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    labels = [m["label"] for m in matches]
    sel = st.selectbox("Elegí partido", labels, index=0)
    m = get_match_by_label(sel)
    if not m:
        st.error("No pude resolver el partido seleccionado.")
        st.stop()

    # Resolver rutas: jugadores (NacSport preferente; si no, TotalValues) + equipo (si existe)
    xml_jug = pr_find_xml_jugadores_for_match(m)
    xml_eq = m.get("xml_equipo")  # puede ser None

    if not xml_jug or not os.path.isfile(xml_jug):
        st.error("No encontré XML NacSport/TotalValues con eventos de jugadores para este partido.")
        st.stop()

    # Cargar
    df_raw = pr_cargar_datos(xml_jug)
    df_pres = pr_cargar_presencias_equipo(xml_eq) if xml_eq else pd.DataFrame(columns=["nombre","rol","start_s","end_s"])
    st.caption(f"Usando eventos de: {os.path.basename(xml_jug)}" + (f" y presencias: {os.path.basename(xml_eq)}" if xml_eq else " (sin XML de equipo)"))

    # Modo de análisis
    modo = st.radio("Modo", ["Equipo", "Jugador"], horizontal=True)
    jugador_sel = None
    if modo == "Jugador":
        nombres = sorted({pr_split_name_role(c)[0] for c in df_raw["jugador"] if pr_split_name_role(c)[0]})
        if not nombres:
            st.warning("No pude inferir nombres de jugadores en el XML.")
            st.stop()
        jugador_sel = st.selectbox("Elegí jugador", nombres, index=0)

    # Opción de acumular todos los partidos
    acum = st.checkbox("🔁 Acumular TODOS los partidos disponibles", value=False)

    def _procesar_un_partido(match_obj):
        xjug = pr_find_xml_jugadores_for_match(match_obj)
        if not xjug or not os.path.isfile(xjug):
            return None
        xeq = match_obj.get("xml_equipo")
        dfr = pr_cargar_datos(xjug)
        dfp = pr_cargar_presencias_equipo(xeq) if xeq else pd.DataFrame(columns=["nombre","rol","start_s","end_s"])
        tot, perd, recu, pperd, precu, dreg = pr_procesar(dfr, dfp if modo=="Jugador" else None, jugador_sel if modo=="Jugador" else None)
        dfres = pr_resumen_df(tot, perd, recu, pperd, precu)
        return {"total": tot, "perd": perd, "recu": recu, "pperd": pperd, "precu": precu, "reg": dreg, "res": dfres}

    if not acum:
        R = _procesar_un_partido(m)
    else:
        agg = None
        for mi in matches:
            Ri = _procesar_un_partido(mi)
            if Ri is None:
                continue
            if agg is None:
                agg = Ri
            else:
                for k in ["total", "perd", "recu", "pperd", "precu"]:
                    agg[k] += Ri[k]  # acumulamos contadores
                agg["reg"] = pd.concat([agg["reg"], Ri["reg"]], ignore_index=True)
        if agg is None:
            st.error("No se pudo acumular: no hay partidos con XML válido.")
            st.stop()
        # Recalcular porcentajes con los acumulados
        with np.errstate(divide='ignore', invalid='ignore'):
            pperd = np.divide(agg["perd"], agg["total"], out=np.zeros_like(agg["perd"]), where=agg["total"] > 0)
            precu = np.divide(agg["recu"], agg["total"], out=np.zeros_like(agg["recu"]), where=agg["total"] > 0)
        agg["pperd"], agg["precu"] = pperd, precu
        agg["res"] = pr_resumen_df(agg["total"], agg["perd"], agg["recu"], pperd, precu)
        R = agg

    if R is None:
        st.error("No hay datos para procesar.")
        st.stop()

    # Tablas + visuals
    st.markdown("#### Resumen por zona")
    st.dataframe(
        R["res"][["zona", "total_acciones", "%_recuperaciones_sobre_total", "%_perdidas_sobre_total"]]
        .rename(columns={
            "%_recuperaciones_sobre_total": "% RECUP.",
            "%_perdidas_sobre_total": "% PÉRD."
        }),
        use_container_width=True
    )

    M_tot = R["total"]
    figR = pr_heatmap(R["precu"], M_tot, f"{'Equipo' if modo=='Equipo' else jugador_sel} — % RECUPERACIONES", good_high=True)
    figP = pr_heatmap(R["pperd"], M_tot, f"{'Equipo' if modo=='Equipo' else jugador_sel} — % PÉRDIDAS", good_high=False)
    st.pyplot(figR, use_container_width=True)
    st.pyplot(figP, use_container_width=True)

    figBR = pr_bars(R["res"], "%_recuperaciones_sobre_total", f"{'Equipo' if modo=='Equipo' else jugador_sel} — Ranking Zonas por % RECUP.")
    figBP = pr_bars(R["res"], "%_perdidas_sobre_total", f"{'Equipo' if modo=='Equipo' else jugador_sel} — Ranking Zonas por % PÉRD.")
    st.pyplot(figBR, use_container_width=True)
    st.pyplot(figBP, use_container_width=True)


# =========================
# UI — 🎯 Mapa de tiros
# =========================
elif menu == "🎯 Mapa de tiros":
    st.subheader("Mapa de tiros — por partido / jugador / rol")

    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    labels = [m["label"] for m in matches]
    sel = st.selectbox("Elegí partido", labels, index=0)
    m = get_match_by_label(sel)
    if not m:
        st.error("No pude resolver el partido seleccionado.")
        st.stop()

    # Preferimos NacSport; si no hubiera, TotalValues
    xml_path = m.get("xml_nacsport") or m.get("xml_players") or m.get("xml_totalvalues")
    if not xml_path or not os.path.isfile(xml_path):
        st.warning("No encontré XML NacSport/TotalValues para este partido.")
        st.stop()

    # ---- Helpers específicos (reutilizan nomenclatura y lógica original) ----
    ANCHO, ALTO = 35.0, 20.0
    N_COLS, N_ROWS = 3, 3
    FLIP_TO_RIGHT = True     # ataque hacia la derecha
    FLIP_VERTICAL = True     # espejo vertical (arriba/abajo)
    GOAL_PULL = 0.60         # “tirón” suave hacia el arco derecho (visual)

    def parse_instances_generic(xml_path):
        root = ET.parse(xml_path).getroot()
        out, all_x, all_y = [], [], []
        for inst in root.findall(".//instance"):
            code = ntext(inst.findtext("code"))
            labels = [nlower(t.text) for t in inst.findall("./label/text")]
            xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
            ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
            if xs and ys:
                all_x += xs; all_y += ys
            out.append({"code": code, "labels": labels, "xs": xs, "ys": ys})
        max_x = float(max(all_x) if all_x else 19)
        max_y = float(max(all_y) if all_y else 34)
        return out, max_x, max_y

    patt_shot = re.compile(r"\b(tiro|remate)\b")

    def is_shot(ev):
        s = nlower(ev["code"])
        return bool(patt_shot.search(s) or any(patt_shot.search(l or "") for l in ev["labels"]))

    def extract_name_role(ev):
        m = _NAME_ROLE_RE.match(ev["code"])
        if m:
            return m.group(1).strip(), m.group(2).strip()
        for l in ev["labels"]:
            if not l: continue
            mx = re.search(r"\(([^)]+)\)", l)
            if mx:
                return None, mx.group(1).strip()
        return None, None

    KEYS = {
        "gol": re.compile(r"\bgol\b"),
        "ataj": re.compile(r"\bataj"),
        "al_arco": re.compile(r"\btiro\s*al\s*arco\b"),
        "bloqueado": re.compile(r"\bbloquead"),
        "desviado": re.compile(r"\bdesviad"),
        "errado": re.compile(r"\berrad"),
        "pifia": re.compile(r"\bpifi"),
    }
    def has(ev, key):
        patt = KEYS[key]
        if patt.search(nlower(ev["code"])):
            return True
        return any(patt.search(l or "") for l in ev["labels"])

    def shot_result_strict(ev):
        if has(ev, "gol"): return "Gol"
        if has(ev, "ataj") or has(ev, "al_arco"): return "Tiro Atajado"
        if has(ev, "bloqueado"): return "Tiro Bloqueado"
        if has(ev, "desviado"): return "Tiro Desviado"
        if has(ev, "errado") or has(ev, "pifia"): return "Tiro Errado - Pifia"
        return "Sin clasificar"

    CHAR_PATTS = [
        ("de Corner (desde Banda)", re.compile(r"corner\s*\(desde\s*banda\)")),
        ("de Corner (centro)", re.compile(r"corner\s*\(centro\)")),
        ("de Jugada (centro)", re.compile(r"jugada\s*\(centro\)")),
        ("de Tiro Libre", re.compile(r"tiro\s*libre")),
        ("de Rebote", re.compile(r"\brebote\b")),
        ("de Lateral", re.compile(r"\blateral\b")),
        ("de Jugada", re.compile(r"\bde\s*jugada\b")),
    ]
    def shot_characteristic(ev):
        s = nlower(ev["code"]) + " " + " ".join(ev["labels"])
        for name, patt in CHAR_PATTS:
            if patt.search(s):
                return name
        return "de Jugada"

    def map_raw_to_pitch(x_raw, y_raw, max_x, max_y, flip=True, pull=0.0, flip_v=False):
        x = (y_raw / max_y) * ANCHO
        y = (x_raw / max_x) * ALTO
        if flip: x = ANCHO - x
        if flip_v: y = ALTO - y
        if pull and 0.0 < pull < 1.0:
            x = x + pull * (ANCHO - x)
        x = float(np.clip(x, 0.0, ANCHO))
        y = float(np.clip(y, 0.0, ALTO))
        return x, y

    def draw_futsal_pitch_grid(ax):
        dx, dy = ANCHO / N_COLS, ALTO / N_ROWS
        ax.set_facecolor("white")
        ax.plot([0, ANCHO], [0, 0], color="black")
        ax.plot([0, ANCHO], [ALTO, ALTO], color="black")
        ax.plot([0, 0], [0, ALTO], color="black")
        ax.plot([ANCHO, ANCHO], [0, ALTO], color="black")
        ax.plot([ANCHO / 2, ANCHO / 2], [0, ALTO], color="black")
        ax.add_patch(Arc((0, ALTO / 2), 8, 12, angle=0, theta1=270, theta2=90, color="black"))
        ax.add_patch(Arc((ANCHO, ALTO / 2), 8, 12, angle=0, theta1=90, theta2=270, color="black"))
        ax.add_patch(plt.Circle((ANCHO / 2, ALTO / 2), 4, color="black", fill=False))
        ax.add_patch(plt.Circle((ANCHO / 2, ALTO / 2), 0.2, color="black"))
        for j in range(N_ROWS):
            for i in range(N_COLS):
                x0, y0 = i * dx, j * dy
                ax.add_patch(Rectangle((x0, y0), dx, dy, linewidth=0.6, edgecolor='gray', facecolor='none'))
                zona = j * N_COLS + i + 1
                ax.text(x0 + dx - 0.4, y0 + dy - 0.4, str(zona), ha='right', va='top', fontsize=9, color='gray')
        ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis('off')

    def collect_shots(xml_path):
        evs, max_x, max_y = parse_instances_generic(xml_path)
        shots = []
        for ev in evs:
            if not is_shot(ev):
                continue

            name, role = extract_name_role(ev)
            code_norm = nlower(ev["code"])
            if any(code_norm.startswith(prefix) for prefix in (
                "categoria - equipo rival", "categoría - equipo rival",
                "tiempo posecion ferro", "tiempo posesion ferro",
                "tiempo posecion rival", "tiempo posesion rival",
                "tiempo no jugado",
            )):
                continue

            if not name and not role:
                continue
            if not (ev["xs"] and ev["ys"]):
                continue

            coords = [
                map_raw_to_pitch(xr, yr, max_x, max_y, flip=FLIP_TO_RIGHT, pull=GOAL_PULL, flip_v=FLIP_VERTICAL)
                for xr, yr in zip(ev["xs"], ev["ys"])
            ]
            origin = coords[0] if len(coords) == 1 else coords[int(np.argmin([c[0] for c in coords]))]
            shots.append({
                "x": origin[0], "y": origin[1],
                "result": shot_result_strict(ev),
                "char": shot_characteristic(ev),
                "player": name or "", "role": role or ""
            })
        return shots

    shots = collect_shots(xml_path)
    if not shots:
        st.info("No se detectaron tiros para este partido.")
        st.stop()

    dfS = pd.DataFrame(shots)
    jugadores = sorted([j for j in dfS["player"].unique() if j])
    roles = sorted([r for r in dfS["role"].unique() if r])
    chars = sorted(dfS["char"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        pick_players = st.multiselect("Jugadores", jugadores, default=jugadores)
    with c2:
        pick_roles = st.multiselect("Roles", roles, default=roles)
    with c3:
        pick_chars = st.multiselect("Característica (origen)", chars, default=chars)

    f = dfS.copy()
    if pick_players: f = f[f["player"].isin(pick_players)]
    if pick_roles:   f = f[f["role"].isin(pick_roles)]
    if pick_chars:   f = f[f["char"].isin(pick_chars)]

    order = ["Gol", "Tiro Atajado", "Tiro Bloqueado", "Tiro Desviado", "Tiro Errado - Pifia", "Sin clasificar"]
    COLORS = {
        "Gol": "#FFD54F", "Tiro Atajado": "#FFFFFF", "Tiro Bloqueado": "#FF5252",
        "Tiro Desviado": "#FF7043", "Tiro Errado - Pifia": "#6B6F76", "Sin clasificar": "#BDBDBD",
    }

    plt.close("all")
    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    draw_futsal_pitch_grid(ax)

    for res in order:
        pts = f.loc[f["result"] == res, ["x", "y"]].to_numpy()
        if pts.size == 0:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        if res == "Gol":
            ax.scatter(xs, ys, s=160, c=COLORS[res], edgecolors="black", linewidths=0.6, zorder=5, label=res)
        elif res == "Tiro Atajado":
            ax.scatter(xs, ys, s=90, c=COLORS[res], edgecolors="black", linewidths=0.6, zorder=4, label=res)
        elif res == "Tiro Bloqueado":
            ax.scatter(xs, ys, s=100, facecolors="none", edgecolors=COLORS[res], linewidths=1.8, zorder=4, label=res)
        elif res == "Tiro Desviado":
            ax.scatter(xs, ys, s=90, facecolors="none", edgecolors=COLORS[res], linewidths=1.6, zorder=3, label=res)
        elif res == "Tiro Errado - Pifia":
            ax.scatter(xs, ys, s=110, marker='x', c=COLORS[res], linewidths=1.8, zorder=3, label=res)
        elif res == "Sin clasificar":
            ax.scatter(xs, ys, s=70, c=COLORS[res], edgecolors="black", linewidths=0.4, zorder=2, label=res)

    sub_players = ", ".join(pick_players) if pick_players else "Todos"
    sub_roles = ", ".join(pick_roles) if pick_roles else "Todos"
    sub_chars = ", ".join(pick_chars) if pick_chars else "Todas"
    ax.set_title(
        f"SHOTS — Origen (punto más lejano) | Jugadores: {sub_players} | Roles: {sub_roles}\n"
        f"Característica: {sub_chars}",
        fontsize=13, pad=6, weight="bold"
    )
    ax.legend(loc="upper left", frameon=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Tablas de conteo + descarga
    c_res = Counter(r for r in f["result"] if r != "Sin clasificar")
    want_order = ["Gol", "Tiro Atajado", "Tiro Bloqueado", "Tiro Desviado", "Tiro Errado - Pifia"]
    results_counts = {k: int(c_res.get(k, 0)) for k in want_order}
    c_char = Counter(f["char"])
    char_counts = dict(sorted(c_char.items(), key=lambda kv: (-kv[1], kv[0])))

    df_results = pd.DataFrame(
        [{"Categoría": k, "Conteo": v, "%": f"{(v / sum(results_counts.values()) * 100) if sum(results_counts.values()) else 0:.1f}%"}
         for k, v in results_counts.items()]
        + [{"Categoría": "TOTAL", "Conteo": sum(results_counts.values()), "%": "100.0%" if sum(results_counts.values()) else "0.0%"}]
    )
    df_chars = pd.DataFrame(
        [{"Categoría": k, "Conteo": v, "%": f"{(v / sum(c_char.values()) * 100) if sum(c_char.values()) else 0:.1f}%"}
         for k, v in char_counts.items()]
        + [{"Categoría": "TOTAL", "Conteo": sum(c_char.values()), "%": "100.0%" if sum(c_char.values()) else "0.0%"}]
    )

    cta1, cta2 = st.columns(2)
    with cta1:
        st.markdown("**Resultados del tiro**")
        st.dataframe(df_results, use_container_width=True)
        st.download_button("⬇️ Descargar resultados (CSV)",
                           df_results.to_csv(index=False).encode("utf-8"),
                           file_name=f"{sel}_shots_resultados.csv", mime="text/csv")
    with cta2:
        st.markdown("**Característica del origen**")
        st.dataframe(df_chars, use_container_width=True)
        st.download_button("⬇️ Descargar características (CSV)",
                           df_chars.to_csv(index=False).encode("utf-8"),
                           file_name=f"{sel}_shots_caracteristicas.csv", mime="text/csv")


# =========================
# UI — 🗺️ Mapa 3x3  (placeholder consistente con tu stack, sin cambiar lógica)
# =========================
elif menu == "🗺️ Mapa 3x3":
    st.info("Mapa 3x3: conecta aquí tu notebook/función existente. Mantengo el menú para consistencia.")


# =========================
# UI — ⚡ Radar  (placeholder consistente)
# =========================
elif menu == "⚡ Radar":
    st.info("Radar: conecta aquí tu notebook/función existente. Mantengo el menú para consistencia.")
