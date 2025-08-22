# app.py — InsightFutsal (Optimizado, MISMA LÓGICA Y RESULTADOS)

# =========================
# IMPORTS
# =========================
import os, re, glob, unicodedata, math
from typing import Optional, Tuple, Dict, List

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
from collections import Counter, defaultdict

# =========================
# CONFIG / ESTILO
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")
st.title("InsightFutsal")

DATA_MINUTOS = "data/minutos"   # XML TotalValues / NacSport
DATA_MATRIX  = "data/matrix"    # Matrix.xlsx / Matrix.csv
BADGE_DIR    = "images/equipos" # ferro.png, <rival>.png
BANNER_DIR   = "images/banner"

# Colores / estilo panel (idénticos)
bg_green   = "#006633"; text_w = "#FFFFFF"
bar_white  = "#FFFFFF"; bar_rival = "#E6EEF2"; bar_rail = "#0F5E29"
star_c     = "#FFD54A"; loser_alpha = 0.35; orange_win = "#FF8F00"

USE_ORANGE_FOR_WIN = True
RAISE_LABELS       = True
BAR_HEIGHT_FACTOR  = 0.36
LABEL_Y_SHIFT_LOW  = 0.60
LABEL_Y_SHIFT_HIGH = 0.37
TRIM_LOGO_BORDERS  = True

BANNER_H   = 0.145
LOGO_W     = 0.118
TITLE_FS   = 32
SUB_FS     = 19

FOOTER_H        = 0.120
FOOTER_LOGO_W   = 0.110
FOOTER_TITLE_FS = 20
FOOTER_SUB_FS   = 14

mpl.rcParams.update({
    "savefig.facecolor": bg_green,
    "figure.facecolor":  bg_green,
    "axes.facecolor":    bg_green,
    "text.color":        text_w,
})

# Orden y tipos (idéntico)
ROW_ORDER = [
    "Posesión %","Pases totales","Pases OK %","Pases último tercio","Pases al área",
    "Tiros","Tiros al arco","Recuperaciones","Duelos ganados","% Duelos ganados",
    "Corners","Faltas","Goles","Asistencias","Pases clave",
]
PERCENT_ROWS = {"Posesión %", "Pases OK %", "% Duelos ganados"}

# =========================
# UTILIDADES STRING
# =========================
def ntext(s):
    if s is None: return ""
    s = unicodedata.normalize("NFD", str(s))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").strip()

def nlower(s): return ntext(s).lower()

# =========================
# ARCHIVOS / PARTIDOS
# =========================
@st.cache_data(show_spinner=False)
def list_matches() -> List[str]:
    """Busca 'Fecha N° - Rival - XML TotalValues.xml' y devuelve 'Fecha N° - Rival'."""
    if not os.path.isdir(DATA_MINUTOS): return []
    pats = glob.glob(os.path.join(DATA_MINUTOS, "Fecha * - * - XML TotalValues.xml"))
    rx = re.compile(r"^Fecha\s*([\d]+)\s*-\s*(.+?)\s*-\s*XML TotalValues\.xml$", re.I)
    labels = []
    for p in sorted(pats):
        base = os.path.basename(p)
        m = rx.match(base)
        if m:
            labels.append(f"Fecha {m.group(1).strip()} - {m.group(2).strip()}")
    return labels

def rival_from_label(label: str) -> str:
    parts = [p.strip() for p in label.split(" - ", 1)]
    return parts[1] if len(parts) == 2 else label

def infer_paths_for_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Arma rutas XML + Matrix (xlsx/csv)."""
    xml_path = os.path.join(DATA_MINUTOS, f"{label} - XML TotalValues.xml")
    mx_xlsx  = os.path.join(DATA_MATRIX,  f"{label} - Matrix.xlsx")
    mx_csv   = os.path.join(DATA_MATRIX,  f"{label} - Matrix.csv")
    matrix_path = mx_xlsx if os.path.isfile(mx_xlsx) else (mx_csv if os.path.isfile(mx_csv) else None)
    return (xml_path if os.path.isfile(xml_path) else None), matrix_path

@st.cache_data(show_spinner=False)
def discover_matches() -> List[Dict]:
    """Prefiere 'XML NacSport' y cae a 'XML TotalValues'. Agrupa por 'Fecha X - Rival'."""
    if not os.path.isdir(DATA_MINUTOS): return []
    pats = glob.glob(os.path.join(DATA_MINUTOS, "* - XML NacSport.xml"))
    pats += glob.glob(os.path.join(DATA_MINUTOS, "* - XML TotalValues.xml"))
    files = sorted(set(pats))

    buckets: Dict[str, List[str]] = {}
    for p in files:
        base = os.path.basename(p)
        label = re.sub(r"(?i)\s*-\s*xml\s*(nacsport|totalvalues)\.xml$", "", base).strip()
        buckets.setdefault(label, []).append(p)

    def pref_key(path: str) -> int:
        b = os.path.basename(path).lower()
        return 0 if "xml nacsport" in b else (1 if "totalvalues" in b else 2)

    matches = []
    for label, paths in buckets.items():
        pick = sorted(paths, key=pref_key)[0]
        rival = (label.split(" - ")[-1] if " - " in label else label).strip()
        mx_xlsx = os.path.join(DATA_MATRIX, f"{label} - Matrix.xlsx")
        mx_csv  = os.path.join(DATA_MATRIX, f"{label} - Matrix.csv")
        matrix_path = mx_xlsx if os.path.isfile(mx_xlsx) else (mx_csv if os.path.isfile(mx_csv) else None)
        matches.append({"label": label, "xml_players": pick, "rival": rival, "matrix_path": matrix_path, "xml_equipo": None})

    def date_key(m):
        mlabel = m["label"].lower()
        mnum = re.search(r"fecha\s*([0-9]+)", mlabel)
        return (0, int(mnum.group(1))) if mnum else (1, m["label"])

    return sorted(matches, key=date_key)

def get_match_by_label(label: str) -> Optional[Dict]:
    for m in discover_matches():
        if m["label"] == label:
            return m
    return None

# =========================
# LOGOS / IMÁGENES
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
    rows = np.where(useful.any(axis=1))[0]; cols = np.where(useful.any(axis=0))[0]
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
    ax.imshow(img, extent=[cx - width/2, cx + width/2, cy - (width*aspect)/2, cy + (width*aspect)/2], zorder=6)

def badge_path_for(name: str) -> Optional[str]:
    nm = name.strip().lower()
    cands = [
        os.path.join(BADGE_DIR, f"{nm}.png"), os.path.join(BADGE_DIR, f"{nm}.jpg"),
        os.path.join(BADGE_DIR, f"{nm.replace(' ','_')}.png"),
        os.path.join(BADGE_DIR, f"{nm.replace(' ','')}.png"),
        os.path.join(BADGE_DIR, f"{nm.replace(' ','-')}.png"),
    ]
    for p in cands:
        if os.path.isfile(p): return p
    return None

# =========================
# CANCHA (UNIFICADA 35x20)
# =========================
ANCHO, ALTO = 35.0, 20.0
N_COLS, N_ROWS = 3, 3

def draw_futsal_pitch_grid(ax):
    dx, dy = ANCHO / N_COLS, ALTO / N_ROWS
    ax.set_facecolor("white")
    # bordes
    ax.plot([0, ANCHO], [0, 0], color="black")
    ax.plot([0, ANCHO], [ALTO, ALTO], color="black")
    ax.plot([0, 0], [0, ALTO], color="black")
    ax.plot([ANCHO, ANCHO], [0, ALTO], color="black")
    # centro
    ax.plot([ANCHO/2, ANCHO/2], [0, ALTO], color="black")
    ax.add_patch(Arc((0, ALTO/2), 8, 12, angle=0, theta1=270, theta2=90, color="black"))
    ax.add_patch(Arc((ANCHO, ALTO/2), 8, 12, angle=0, theta1=90, theta2=270, color="black"))
    ax.add_patch(MplCircle((ANCHO/2, ALTO/2), 4, color="black", fill=False))
    ax.add_patch(MplCircle((ANCHO/2, ALTO/2), 0.2, color="black"))
    # grilla 3x3
    for j in range(N_ROWS):
        for i in range(N_COLS):
            x0, y0 = i * dx, j * dy
            ax.add_patch(Rectangle((x0, y0), dx, dy, linewidth=0.6, edgecolor='gray', facecolor='none'))
            zona = j * N_COLS + i + 1
            ax.text(x0 + dx - 0.4, y0 + dy - 0.4, str(zona), ha='right', va='top', fontsize=9, color='gray')
    ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis('off')

# =========================
# PARSERS / XML
# =========================
_NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")

_EXCLUDE_PREFIXES = tuple([  # normalizados a lower sin tildes
    "categoria - equipo rival",
    "tiempo posecion ferro", "tiempo posesion ferro",
    "tiempo posecion rival", "tiempo posesion rival",
    "tiempo no jugado",
])

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def is_player_code(code: str) -> bool:
    if not code: return False
    code_norm = _strip_accents(code).lower().strip()
    if any(code_norm.startswith(pref) for pref in _EXCLUDE_PREFIXES): return False
    return _NAME_ROLE_RE.match(code) is not None

def _split_name_role(code: str):
    m = _NAME_ROLE_RE.match(str(code)) if code else None
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)

@st.cache_data(show_spinner=False)
def cargar_datos_nacsport(xml_path: str) -> pd.DataFrame:
    """Lee XML (TotalValues/NacSport) y devuelve DF: jugador, labels, pos_x_list, pos_y_list."""
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["jugador","labels","pos_x_list","pos_y_list"])
    root = ET.parse(xml_path).getroot()
    data = []
    for inst in root.findall(".//instance"):
        jugador = inst.findtext("code") or ""
        labels = [ntext(lbl.findtext("text")) for lbl in inst.findall("label")]
        pos_x = [float(px.text) for px in inst.findall("pos_x") if (px.text or "").strip()!=""]
        pos_y = [float(py.text) for py in inst.findall("pos_y") if (py.text or "").strip()!=""]
        data.append({"jugador": jugador, "labels": labels, "pos_x_list": pos_x, "pos_y_list": pos_y})
    return pd.DataFrame(data)

# ------ para estadísticas de partido ------
def parse_possession_from_equipo(xml_path: str) -> Tuple[float, float]:
    """Posesión desde códigos 'tiempo posecion ferro/rival'."""
    if not xml_path or not os.path.isfile(xml_path): return 0.0, 0.0
    root = ET.parse(xml_path).getroot()
    t_ferro = t_rival = 0.0
    for inst in root.findall(".//instance"):
        code = nlower(inst.findtext("code"))
        try:
            stt = float(inst.findtext("start") or "0")
            enn = float(inst.findtext("end") or "0")
        except:
            continue
        dur = max(0.0, enn - stt)
        if code == "tiempo posecion ferro":  t_ferro += dur
        elif code == "tiempo posecion rival": t_rival += dur
    tot = t_ferro + t_rival
    if tot <= 0: return 0.0, 0.0
    return round(100*t_ferro/tot,1), round(100*t_rival/tot,1)

def load_matrix(path: str) -> Tuple[pd.DataFrame, str, Dict[str,str]]:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path, header=0)
    elif path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path, header=0)
        except Exception:
            df = pd.read_csv(path, header=0, sep=";")
    else:
        raise ValueError("Formato no soportado para MATRIX")
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
        extent=[x_center - size/2, x_center + size/2,
                y_center - (size*asp)/2, y_center + (size*asp)/2],
        zorder=9
    )

from matplotlib.patches import Circle
def draw_count_circle(ax, x, y, count, base_r=0.006, face=None, edge=text_w, lw=1.0, z=8):
    if count <= 0: return
    r = base_r * (count ** 0.25)  # misma curva visual
    circ = Circle((x, y), radius=r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(circ)

def parse_instances_jugadores(xml_path: str):
    if not xml_path or not os.path.isfile(xml_path): return []
    root = ET.parse(xml_path).getroot()
    out = []
    for inst in root.findall(".//instance"):
        code = ntext(inst.findtext("code"))
        try: stt = float(inst.findtext("start") or "0")
        except: stt = 0.0
        try: enn = float(inst.findtext("end") or "0")
        except: enn = stt
        labels_lc = [nlower(t.text) for t in inst.findall("./label/text")]
        xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
        ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
        x_end, y_end = (xs[-1], ys[-1]) if xs and ys else (None, None)
        out.append({"code": code, "labels_lc": labels_lc, "start": stt, "end": enn, "end_xy": (x_end, y_end)})
    return out

def is_rival_code(code) -> bool:
    return nlower(code).startswith("categoria - equipo rival")

def is_pass_attempt(ev) -> bool:
    if re.match(r"^\s*pase\b", nlower(ev["code"])): return True
    return any(re.match(r"^\s*pase\b", l) for l in ev["labels_lc"])

def is_shot(ev):
    s = nlower(ev["code"])
    return bool(re.search(r"\btiro\b|\bremate\b", s) or any(re.search(r"\btiro\b|\bremate\b", l) for l in ev["labels_lc"]))

def is_goal(ev):
    s = nlower(ev["code"])
    return bool(re.match(r"^gol\b", s) or any(re.match(r"^gol\b", l) for l in ev["labels_lc"]))

ON_TARGET_PAT = re.compile(r"\b(al\s*arco|a\s*puerta|a\s*porter[ií]a|on\s*target|atajad[oa]|saved\s*shot)\b", re.I)
def is_shot_on_target(ev):
    s = nlower(ev["code"])
    return bool(ON_TARGET_PAT.search(s) or any(ON_TARGET_PAT.search(l or "") for l in ev["labels_lc"]))

def minute_bucket(sec):
    m = int(sec // 60)
    return max(0, min(39, m))  # 0..39

def xy_to_zone(x, y, max_x=19, max_y=34):
    if x is None or y is None: return None
    col = 1 if x <= 6 else (2 if x <= 13 else 3)
    row = 1 if y <= 11 else (2 if y <= 22 else 3)
    return (row-1)*3 + col  # 1..9

def passes_last_third_and_area(jug):
    last_m = last_r = area_m = area_r = 0
    for ev in jug:
        if not is_pass_attempt(ev): continue
        z = xy_to_zone(*ev["end_xy"])
        if z is None: continue
        if is_rival_code(ev["code"]):
            if z in {1,4,7}: last_r += 1
            if z == 4: area_r += 1
        else:
            if z in {3,6,9}: last_m += 1
            if z == 6: area_m += 1
    return last_m, last_r, area_m, area_r

def build_timeline(xml_players_path):
    evs = parse_instances_jugadores(xml_players_path)
    M = 40
    tl = dict(
        passes_M=np.zeros(M, int), passes_R=np.zeros(M, int),
        last_M=np.zeros(M, int),   last_R=np.zeros(M, int),
        shots_on_M=np.zeros(M, int), shots_off_M=np.zeros(M, int),
        shots_on_R=np.zeros(M, int), shots_off_R=np.zeros(M, int),
        goals_M=np.zeros(M, int),  goals_R=np.zeros(M, int),
    )
    for ev in evs:
        m = minute_bucket(ev.get("end", ev.get("start", 0.0)))
        if is_pass_attempt(ev):
            if is_rival_code(ev["code"]): tl["passes_R"][m] += 1
            else:                          tl["passes_M"][m] += 1
            z = xy_to_zone(*(ev["end_xy"] or (None, None)))
            if z is not None:
                if is_rival_code(ev["code"]):
                    if z in {1,4,7}: tl["last_R"][m] += 1
                else:
                    if z in {3,6,9}: tl["last_M"][m] += 1
        if is_shot(ev):
            goal = is_goal(ev); on_t = is_shot_on_target(ev) or goal
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
yellow_on  = "#FFD54F"; rival_g = "#B9C4C9"; white = "#FFFFFF"

def draw_timeline_panel(rival_name: str, tl: dict,
                        ferro_logo_path: Optional[str], rival_logo_path: Optional[str]):
    plt.close("all")
    fig_h = 11.8
    fig = plt.figure(figsize=(10.8, fig_h))
    ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0), 1, 1, facecolor=bg_green, edgecolor="none"))

    # Banner sup
    BANNER_H_loc = 0.14; BANNER_Y0 = 1 - BANNER_H_loc
    ax.add_patch(Rectangle((0, BANNER_Y0), 1, BANNER_H_loc, facecolor=white, edgecolor="none"))
    if ferro_logo_path: draw_logo(ax, ferro_logo_path, 0.075, BANNER_Y0+0.07, 0.12)
    if rival_logo_path: draw_logo(ax, rival_logo_path, 0.925, BANNER_Y0+0.07, 0.12)
    ax.text(0.5, BANNER_Y0+0.085, f"FERRO vs {rival_name.upper()}", ha="center", va="center",
            color=bg_green, fontsize=30, weight="bold")
    ax.text(0.5, BANNER_Y0+0.040, "TIMELINE", ha="center", va="center",
            color=bg_green, fontsize=16, weight="bold")

    # Banner inferior
    FOOT_H, FOOT_Y0 = 0.12, 0.00
    ax.add_patch(Rectangle((0, FOOT_Y0), 1, FOOT_H, facecolor=white, edgecolor="none"))
    if os.path.isfile(FOOTER_LEFT_LOGO):  draw_logo(ax, FOOTER_LEFT_LOGO,  0.09, FOOT_Y0+FOOT_H*0.52, 0.14)
    if os.path.isfile(FOOTER_RIGHT_LOGO): draw_logo(ax, FOOTER_RIGHT_LOGO, 0.91, FOOT_Y0+FOOT_H*0.52, 0.12)
    ax.text(0.50, FOOT_Y0+FOOT_H*0.62, "TRABAJO FIN DE MÁSTER", ha="center", va="center",
            color=bg_green, fontsize=18, weight="bold")
    ax.text(0.50, FOOT_Y0+FOOT_H*0.32, "Cristian Dieguez", ha="center", va="center",
            color=bg_green, fontsize=13, weight="bold")

    # Panel central
    panel_y0, panel_y1 = FOOT_Y0+FOOT_H+0.05, BANNER_Y0-0.05
    panel_h = panel_y1 - panel_y0

    x_center_gap_L, x_center_gap_R = 0.47, 0.53
    x_bar_M_max, x_bar_R_max = 0.22, 0.78
    x_shot_M, x_last_M = 0.05, 0.16
    x_shot_R, x_last_R = 0.95, 0.84
    x_goal_M, x_goal_R = x_shot_M - 0.025, x_shot_R + 0.025

    # Títulos
    ty = panel_y1 + 0.012
    ax.text(x_last_M,  ty, "Últ. tercio", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_bar_M_max+x_center_gap_L)/2, ty, "Pases/min", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_shot_M,  ty, "Tiros / Goles", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_center_gap_L+x_center_gap_R)/2, ty, "Min.", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_center_gap_R+x_bar_R_max)/2, ty, "Pases/min", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_shot_R,  ty, "Tiros / Goles", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_last_R,  ty, "Últ. tercio", ha="center", va="bottom", fontsize=10, weight="bold")

    # Minutos 0..40
    M = 40
    def y_of_index(i): return panel_y1 - panel_h * (i + 0.5) / M

    ax.add_line(plt.Line2D([x_center_gap_L, x_center_gap_L], [panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    ax.add_line(plt.Line2D([x_center_gap_R, x_center_gap_R], [panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    for m in range(0, 41, 5):
        yy = y_of_index(min(m, 39))
        ax.text(0.50, yy, f"{m:02d}'", ha="center", va="center", fontsize=8, alpha=0.90)

    max_bar = max(int(tl["passes_M"].max() if tl["passes_M"].size else 1),
                  int(tl["passes_R"].max() if tl["passes_R"].size else 1), 1)

    def bar_width_left(cnt):
        return (x_center_gap_L - x_bar_M_max) * (cnt / max_bar if max_bar else 0)

    def bar_width_right(cnt):
        return (x_bar_R_max - x_center_gap_R) * (cnt / max_bar if max_bar else 0)

    bar_h = panel_h / M * 0.55

    for m in range(M):
        y = y_of_index(m)
        # izquierda (FERRO)
        wL = bar_width_left(int(tl["passes_M"][m])); x0L = x_center_gap_L - wL
        ax.add_patch(Rectangle((x0L, y - bar_h/2), wL, bar_h, facecolor=orange_win, edgecolor="none"))
        if tl["passes_M"][m] > 0:
            ax.text(x0L - 0.006, y, f"{tl['passes_M'][m]}", ha="right", va="center", fontsize=8)

        draw_count_circle(ax, x_last_M, y, int(tl["last_M"][m]), base_r=0.006, face=white, edge=white, lw=0.0, z=7)
        draw_count_circle(ax, x_shot_M, y, int(tl["shots_off_M"][m]), base_r=0.006, face=None,   edge=white, lw=1.0, z=8)
        draw_count_circle(ax, x_shot_M, y, int(tl["shots_on_M"][m]),  base_r=0.006, face=yellow_on, edge=white, lw=0.6, z=9)
        if tl["goals_M"][m] > 0:
            for k in range(int(tl["goals_M"][m])):
                draw_ball(ax, x_goal_M - k*0.012, y, size=0.016)

        # derecha (RIVAL)
        wR = bar_width_right(int(tl["passes_R"][m])); x0R = x_center_gap_R
        ax.add_patch(Rectangle((x0R, y - bar_h/2), wR, bar_h, facecolor=rival_g, edgecolor="none"))
        if tl["passes_R"][m] > 0:
            ax.text(x0R + wR + 0.006, y, f"{tl['passes_R'][m]}", ha="left", va="center", fontsize=8)

        draw_count_circle(ax, x_last_R, y, int(tl["last_R"][m]), base_r=0.006, face=white, edge=white, lw=0.0, z=7)
        draw_count_circle(ax, x_shot_R, y, int(tl["shots_off_R"][m]), base_r=0.006, face=None,   edge=white, lw=1.0, z=8)
        draw_count_circle(ax, x_shot_R, y, int(tl["shots_on_R"][m]),  base_r=0.006, face=yellow_on, edge=white, lw=0.6, z=9)
        if tl["goals_R"][m] > 0:
            for k in range(int(tl["goals_R"][m])):
                draw_ball(ax, x_goal_R + k*0.012, y, size=0.016)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================
# HEATMAPS
# =========================
KEYWORDS_PASES = [k.lower() for k in [
    "pase corto frontal","pase corto lateral","pase largo frontal","pase largo lateral",
    "pase progresivo frontal","pase progresivo lateral",
    "pase progresivo frontal cpie","pase progresivo lateral cpie",
    "salida de arco progresivo cmano","pase corto frontal cpie","pase corto lateral cpie",
    "salida de arco corto cmano"
]]

def _es_evento_pase_o_tiro(labels: List[str]) -> bool:
    lbls = [nlower(l or "") for l in labels]
    is_shot = any(("tiro" in l or "remate" in l) for l in lbls)
    is_pass = any(any(k in l for k in KEYWORDS_PASES) for l in lbls)
    return is_shot or is_pass

def filtrar_coords_por_evento(row: pd.Series) -> Tuple[List[float], List[float]]:
    if _es_evento_pase_o_tiro(row.get("labels", [])):
        return row["pos_x_list"][:1], row["pos_y_list"][:1]
    return row["pos_x_list"], row["pos_y_list"]

def parse_player_and_role(code: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.*)\((.*)\)\s*$", code.strip())
    if m:
        return ntext(m.group(1)).strip(), ntext(m.group(2)).strip()
    return code.strip(), None

def explode_coords_for_heatmap(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty: return pd.DataFrame(columns=["player_name","role","pos_x","pos_y"])
    tmp = df_raw.copy()
    tmp["pos_x_list"], tmp["pos_y_list"] = zip(*tmp.apply(filtrar_coords_por_evento, axis=1))
    tmp = tmp.explode(["pos_x_list","pos_y_list"], ignore_index=True)
    tmp = tmp.rename(columns={"pos_x_list":"pos_x","pos_y_list":"pos_y"}).dropna(subset=["pos_x","pos_y"])

    names_roles = tmp["jugador"].apply(parse_player_and_role)
    tmp["player_name"] = names_roles.apply(lambda t: t[0]).map(ntext)
    tmp["role"]        = names_roles.apply(lambda t: t[1]).map(lambda x: ntext(x) if x else None)

    return tmp[["player_name","role","pos_x","pos_y"]]

def rotate_coords_for_attack_right(df: pd.DataFrame, role: Optional[str]=None) -> pd.DataFrame:
    """Rota coords del XML (20x35) a cancha horizontal (35x20). Invierte Y para Ala I / Ala D."""
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
    draw_futsal_pitch_grid(ax)
    if df_xy.empty:
        ax.text(0.5, 0.5, "Sin datos para el filtro", ha="center", va="center", transform=ax.transAxes)
        return fig
    pastel_cmap = LinearSegmentedColormap.from_list("pastel_heatmap", ["#a8dadc", "#bde0c6", "#ffe5a1", "#f6a192"])
    sns.kdeplot(x=df_xy["x_rot"], y=df_xy["y_rot"], fill=True, cmap=pastel_cmap, bw_adjust=0.4,
                levels=50, alpha=0.75, ax=ax, clip=((0, 35), (0, 20)))
    ax.set_title(titulo, fontsize=14)
    return fig

# =========================
# MINUTOS
# =========================
def _merge_intervals(intervals):
    ints = [(float(s), float(e)) for s, e in intervals if s is not None and e is not None and e > s]
    if not ints: return []
    ints.sort(); merged = [list(ints[0])]
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
        stt = inst.findtext("start"); enn = inst.findtext("end")
        try:
            s = float(stt) if stt is not None else None
            e = float(enn) if enn is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s: continue
        rows.append({"code": code, "nombre": nombre, "rol": rol, "start_s": s, "end_s": e, "dur_s": e - s})
    return pd.DataFrame(rows)

def _format_mmss(seconds: float | int) -> str:
    if seconds is None or not np.isfinite(seconds): return "00:00"
    s = int(round(float(seconds))); mm, ss = divmod(s, 60)
    return f"{mm:02d}:{ss:02d}"

def minutos_por_presencia(df_pres: pd.DataFrame):
    if df_pres.empty:
        empt1 = pd.DataFrame(columns=["nombre","rol","segundos","mmss","minutos","n_tramos"])
        empt2 = pd.DataFrame(columns=["nombre","segundos","mmss","minutos","n_tramos"])
        return (empt1, empt2)
    out = []
    for (nombre, rol), g in df_pres.groupby(["nombre","rol"], dropna=False):
        intervals = list(zip(g["start_s"], g["end_s"]))
        merged = _merge_intervals(intervals)
        secs = sum(e - s for s, e in merged)
        out.append({"nombre": nombre,"rol": rol,"segundos": int(round(secs)),
                    "mmss": _format_mmss(secs),"minutos": secs/60.0,"n_tramos": len(merged)})
    df_por_rol = (pd.DataFrame(out).sort_values(["segundos","nombre"], ascending=[False, True]).reset_index(drop=True))
    df_por_rol["minutos"] = df_por_rol["minutos"].round(2)
    df_por_jugador = (df_por_rol.groupby("nombre", as_index=False)
                      .agg(segundos=("segundos","sum"), minutos=("minutos","sum"), n_tramos=("n_tramos","sum"))
                      .assign(mmss=lambda d: d["segundos"].apply(_format_mmss))
                      .sort_values(["segundos","nombre"], ascending=[False, True]).reset_index(drop=True))
    df_por_jugador["minutos"] = df_por_jugador["minutos"].round(2)
    return df_por_rol, df_por_jugador

def _inside_bar_label(ax, bars, values_sec):
    xlim = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else max(1, (np.array(values_sec)/60).max() if len(values_sec) else 1)
    for b, secs in zip(bars, values_sec):
        mmss = _format_mmss(secs)
        x = b.get_width(); y = b.get_y() + b.get_height()/2
        txt_x = x * 0.98
        if x < 0.9:
            txt_x = min(x + 0.1, xlim * 0.98); ha = "left"; color = "black"; fw="bold"
        else:
            ha = "right"; color = "white"; fw="bold"
        ax.text(txt_x, y, mmss, va="center", ha=ha, fontsize=9, color=color, fontweight=fw)

def fig_barh_minutos(labels, vals_sec, title, xlabel="Minutos"):
    vals_min = (np.array(vals_sec) / 60.0)
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.48*len(labels))))
    bars = ax.barh(labels, vals_min, alpha=0.9); ax.invert_yaxis()
    _inside_bar_label(ax, bars, vals_sec)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    vmax = max(vals_min) if len(vals_min) else 1.0
    ax.set_xlim(0, vmax * 1.12 + 0.4); plt.tight_layout()
    return fig

# =========================
# RED DE PASES — POR ROL
# =========================
PASS_KEYWORDS = KEYWORDS_PASES  # mismos keywords

def red_de_pases_por_rol(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_futsal_pitch_grid(ax)

    coords_origen = defaultdict(list)
    totales_hechos_por_rol = defaultdict(int)
    conteo_roles_total = defaultdict(int)

    for _, row in df.iterrows():
        code = row.get("jugador") or ""
        if not is_player_code(code): continue
        labels_lower = [(_strip_accents(lbl).lower() if lbl else "") for lbl in row.get("labels", [])]
        if not any(k in lbl for lbl in labels_lower for k in PASS_KEYWORDS): continue

        rol_origen = _NAME_ROLE_RE.match(code).group(2).strip()
        rol_destino = None
        for lbl in row.get("labels", []):
            if lbl and "(" in lbl and ")" in lbl:
                pos = lbl.split("(")[1].replace(")", "").strip()
                if pos and pos != rol_origen:
                    rol_destino = pos; break

        px, py = row.get("pos_x_list") or [], row.get("pos_y_list") or []
        if px and py:
            x0 = 35 - (py[0] * (35.0 / 40.0))  # y original (0..40) -> x cancha 35
            y0 = px[0]                         # x original (0..20) -> y cancha 20
            coords_origen[rol_origen].append((x0, y0))

        totales_hechos_por_rol[rol_origen] += 1
        if rol_destino and rol_destino != rol_origen:
            conteo_roles_total[tuple(sorted([rol_origen, rol_destino]))] += 1

    # promedio por rol
    rol_coords = {}
    for rol, coords in coords_origen.items():
        arr = np.array(coords)
        rol_coords[rol] = (arr[:,0].mean(), arr[:,1].mean())

    # Ajustes heurísticos y swap si Ala I/D quedan invertidos
    if "Arq" in rol_coords: rol_coords["Arq"] = (3, 10)
    if "Ala I" in rol_coords and "Ala D" in rol_coords and rol_coords["Ala I"][1] < rol_coords["Ala D"][1]:
        rol_coords["Ala I"], rol_coords["Ala D"] = rol_coords["Ala D"], rol_coords["Ala I"]
        new_counts = defaultdict(int)
        for (a,b), v in conteo_roles_total.items():
            aa = "Ala D" if a=="Ala I" else ("Ala I" if a=="Ala D" else a)
            bb = "Ala D" if b=="Ala I" else ("Ala I" if b=="Ala D" else b)
            new_counts[tuple(sorted([aa,bb]))] += v
        conteo_roles_total = new_counts
        new_tot = defaultdict(int)
        for r,t in totales_hechos_por_rol.items():
            if r=="Ala I": new_tot["Ala D"] = t
            elif r=="Ala D": new_tot["Ala I"] = t
            else: new_tot[r] = t
        totales_hechos_por_rol = new_tot

    # edges
    max_pases = max(conteo_roles_total.values()) if conteo_roles_total else 1
    for (ra, rb), count in conteo_roles_total.items():
        if ra in rol_coords and rb in rol_coords:
            x1,y1 = rol_coords[ra]; x2,y2 = rol_coords[rb]
            lw = 1 + (count / max_pases) * 5
            ax.plot([x1,x2], [y1,y2], color='red', linewidth=lw, alpha=0.7, zorder=3)
            ax.text((x1+x2)/2, (y1+y2)/2, str(count), color='blue', fontsize=10,
                    ha='center', va='center', fontweight='bold', zorder=6)

    # nodos
    for rol, (x,y) in rol_coords.items():
        ax.scatter(x, y, s=2000, color='white', edgecolors='black', zorder=5)
        ax.text(x, y, f"{rol}\n{totales_hechos_por_rol.get(rol,0)}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='black', zorder=7)

    ax.set_title("Red de Pases por Rol (NacSport/TotalValues; sólo Jugador (Rol))", fontsize=13)
    plt.tight_layout()
    return fig

# =========================
# PÉRDIDAS & RECUPERACIONES
# =========================
PR_ANCHO, PR_ALTO = 35, 20
PR_N_COLS, PR_N_ROWS = 3, 3
_PR_NAME_ROLE_RE = _NAME_ROLE_RE
_PR_EXCLUDE_PREFIXES = _EXCLUDE_PREFIXES

def pr_strip_accents(s: str) -> str: return _strip_accents(s)
def pr_is_player_code(code: str) -> bool: return is_player_code(code)
def pr_split_name_role(code: str): return _split_name_role(code)

def pr_cargar_datos(xml_path: str) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    data = []
    for inst in root.findall(".//instance"):
        jugador = inst.findtext("code") or ""
        if not pr_is_player_code(jugador): continue
        pos_x = [float(px.text) for px in inst.findall("pos_x") if (px.text or "").strip()!=""]
        pos_y = [float(py.text) for py in inst.findall("pos_y") if (py.text or "").strip()!=""]
        labels = [ (lbl.findtext("text") or "").strip() for lbl in inst.findall("label") ]
        stt = inst.findtext("start"); enn = inst.findtext("end")
        try:
            start = float(stt) if stt is not None else None
            end   = float(enn) if enn is not None else None
        except Exception:
            start, end = None, None
        data.append({"jugador": jugador,"labels": [l for l in labels if l],
                     "pos_x_list": pos_x,"pos_y_list": pos_y,"start": start,"end": end})
    return pd.DataFrame(data)

def pr_cargar_presencias_equipo(xml_path: str) -> pd.DataFrame:
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["nombre","rol","start_s","end_s"])
    root = ET.parse(xml_path).getroot()
    rows = []
    for inst in root.findall(".//instance"):
        code = inst.findtext("code") or ""
        nombre, rol = pr_split_name_role(code)
        if not nombre: continue
        stt = inst.findtext("start"); enn = inst.findtext("end")
        try:
            s = float(stt) if stt is not None else None
            e = float(enn) if enn is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s: continue
        rows.append({"nombre": nombre, "rol": rol, "start_s": s, "end_s": e})
    return pd.DataFrame(rows)

def pr_labels_low(labels): return [pr_strip_accents(l).lower() for l in (labels or [])]
def pr_transform_xy(px, py):  return PR_ANCHO - (py * (PR_ANCHO / 40.0)), px
def pr_rol_de(code):
    if code and "(" in code and ")" in code:
        return code.split("(")[1].split(")")[0].strip()
    return None

def pr_ajustar_ala(y, code):
    if pr_rol_de(code) in ("Ala I","Ala D"): return PR_ALTO - y
    return y

def pr_zona_from_xy(x, y):
    dx, dy = PR_ANCHO / PR_N_COLS, PR_ALTO / PR_N_ROWS
    col = min(int(x // dx), PR_N_COLS - 1)
    row = min(int(y // dy), PR_N_ROWS - 1)
    return row, col

def pr_zona_id(row, col): return row * PR_N_COLS + col + 1
def pr_get_point(pxs, pys, idx):
    if not pxs: return None
    idx = max(0, min(idx, len(pxs)-1))
    return pxs[idx], pys[idx]

_PR_KW_TOTAL   = [k.lower() for k in ["pase","recuperacion","recuperación","perdida","pérdida","pierde","conseguido","faltas","centro","lateral","despeje","despeje rival","gira","aguanta","cpie","cmano"]]
_PR_KW_PERDIDA = [k.lower() for k in ["perdida","pérdida","pierde","despeje"]]
_PR_KW_RECU    = [k.lower() for k in ["recuperacion","recuperación","1v1 ganado","despeje rival","cpie","cmano"]]

def pr_contiene(labels_low, kws): return any(k in l for l in labels_low for k in kws)

def pr_zone_for(evento, code, pxs, pys):
    idx = 1 if (evento=="PERDIDA" and pxs and len(pxs)>1) else 0
    p = pr_get_point(pxs, pys, idx)
    if p is None: return None
    x, y = pr_transform_xy(*p); y = pr_ajustar_ala(y, code)
    return pr_zona_from_xy(x, y)

def pr_merge_minutes(df_pres: pd.DataFrame) -> dict:
    out = {}
    for (n,_), g in df_pres.groupby(["nombre","rol"], dropna=False):
        ints = [(float(s), float(e)) for s,e in zip(g["start_s"], g["end_s"]) if pd.notna(s) and pd.notna(e) and e> s]
        ints.sort(); merged=[]
        for s,e in ints:
            if not merged or s>merged[-1][1]: merged.append([s,e])
            else: merged[-1][1]=max(merged[-1][1], e)
        out.setdefault(n, []).extend([(s,e) for s,e in merged])
    return out

def pr_on_court(name_intervals, t):
    if t is None: return False
    for s,e in name_intervals:
        if s<=t<e: return True
    return False

def pr_procesar(df_raw: pd.DataFrame, df_pres: pd.DataFrame|None, jugador_filter: str|None):
    name2ints = pr_merge_minutes(df_pres) if df_pres is not None and not df_pres.empty else {}
    total_acc = np.zeros((PR_N_ROWS, PR_N_COLS), dtype=float)
    perdidas  = np.zeros_like(total_acc)
    recupera  = np.zeros_like(total_acc)
    registros = []

    for _, r in df_raw.iterrows():
        code = r["jugador"]; labels_low = pr_labels_low(r["labels"])
        if jugador_filter:
            nombre, _ = pr_split_name_role(code)
            if not nombre or jugador_filter.lower() not in nombre.lower(): continue
            if name2ints.get(nombre) and not pr_on_court(name2ints[nombre], r.get("start")): continue

        pxs, pys = r["pos_x_list"], r["pos_y_list"]
        if not pxs: continue

        has_perd = pr_contiene(labels_low, _PR_KW_PERDIDA)
        has_recu = pr_contiene(labels_low, _PR_KW_RECU)
        de_interes = pr_contiene(labels_low, _PR_KW_TOTAL)

        counted = False
        if has_perd:
            z = pr_zone_for("PERDIDA", code, pxs, pys)
            if z:
                rr,cc = z; perdidas[rr,cc]+=1; total_acc[rr,cc]+=1; counted=True
                registros.append((rr,cc,pr_zona_id(rr,cc), code, labels_low, "PÉRDIDA", 1.0, r.get("start")))
        if has_recu:
            z = pr_zone_for("RECUPERACION", code, pxs, pys)
            if z:
                rr,cc = z; recupera[rr,cc]+=1; total_acc[rr,cc]+=1; counted=True
                registros.append((rr,cc,pr_zona_id(rr,cc), code, labels_low, "RECUPERACIÓN", 1.0, r.get("start")))
        if (not counted) and de_interes:
            p = pr_get_point(pxs, pys, 0)
            if p is None: continue
            x,y = pr_transform_xy(*p); y = pr_ajustar_ala(y, code)
            rr,cc = pr_zona_from_xy(x,y)
            total_acc[rr,cc]+=1
            registros.append((rr,cc,pr_zona_id(rr,cc), code, labels_low, "OTROS", 1.0, r.get("start")))

    with np.errstate(divide='ignore', invalid='ignore'):
        porc_perd = np.divide(perdidas, total_acc, out=np.zeros_like(perdidas), where=total_acc>0)
        porc_recu = np.divide(recupera, total_acc, out=np.zeros_like(recupera), where=total_acc>0)

    df_reg = pd.DataFrame(registros, columns=["row","col","zona","jugador","labels_low","tipo","peso","t0"])
    return total_acc, perdidas, recupera, porc_perd, porc_recu, df_reg

def pr_draw_pitch_grid(ax): draw_futsal_pitch_grid(ax)

def pr_heatmap(matriz_pct, matriz_tot, title, good_high=True):
    fig, ax = plt.subplots(figsize=(9, 6))
    pr_draw_pitch_grid(ax)
    cmap = (LinearSegmentedColormap.from_list("good", ["#f0f9e8","#bae4bc","#7bccc4","#2b8cbe","#08589e"])
            if good_high else
            LinearSegmentedColormap.from_list("bad",  ["#fff5f0","#fcbba1","#fc9272","#ef3b2c","#99000d"]))
    dx, dy = PR_ANCHO / PR_N_COLS, PR_ALTO / PR_N_ROWS
    for j in range(PR_N_ROWS):
        for i in range(PR_N_COLS):
            v = float(matriz_pct[j,i]); tot=float(matriz_tot[j,i])
            color = cmap(v) if tot>0 else (0.95,0.95,0.95,1.0)
            ax.add_patch(Rectangle((i*dx,j*dy), dx,dy, color=color, alpha=0.7))
            if tot>0:
                ax.text(i*dx+dx/2, j*dy+dy/2, f"{v*100:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(i*dx+dx-0.6, j*dy+0.35, f"Tot: {int(round(tot))}", ha="right", va="bottom", fontsize=9, color="black")
    plt.title(title); plt.tight_layout()
    return fig

def pr_bars(df, col_pct, title):
    d = (df[["zona", col_pct, "total_acciones"]].rename(columns={col_pct:"pct"})).copy()
    d["zona_lbl"] = d["zona"].apply(lambda z: f"Z{int(z)}")
    d = d.sort_values("pct", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["zona_lbl"], d["pct"], height=0.6); ax.invert_yaxis()
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax.set_xlabel("%"); ax.set_title(title); ax.grid(axis="x", linestyle=":", alpha=0.35)
    vmax = max(0.01, d["pct"].max()) + 0.15
    for y, (v, t) in enumerate(zip(d["pct"], d["total_acciones"])):
        ax.text(min(v+0.01, vmax-0.02), y, f"{v*100:.1f}%  (Tot {int(round(t))})", va="center", fontsize=9)
    ax.set_xlim(0, vmax); plt.tight_layout()
    return fig

def pr_mat_from_df(df, col):
    M = np.zeros((PR_N_ROWS, PR_N_COLS), dtype=float)
    for _, r in df.iterrows():
        M[int(r["row"]), int(r["col"])] = float(r[col])
    return M

def pr_resumen_df(total_acc, perdidas, recupera, porc_perd, porc_recu):
    rows=[]
    for r in range(PR_N_ROWS):
        for c in range(PR_N_COLS):
            t = float(total_acc[r,c]); p=float(perdidas[r,c]); rc=float(recupera[r,c])
            rows.append({"zona":r*PR_N_COLS+c+1,"row":r,"col":c,
                         "total_acciones":t,
                         "%_perdidas_sobre_total": (p/t if t else 0.0),
                         "%_recuperaciones_sobre_total": (rc/t if t else 0.0)})
    return pd.DataFrame(rows).sort_values("zona").reset_index(drop=True)

def pr_find_xml_jugadores_for_match(match_obj):
    if match_obj.get("xml_jugadores") and os.path.isfile(match_obj["xml_jugadores"]):
        return match_obj["xml_jugadores"]
    rival = match_obj.get("rival") or ""
    cands = []
    for pat in [f"*{rival}*XML NacSport*.xml", f"*{rival}*asXML TotalValues*.xml"]:
        cands += glob.glob(os.path.join(DATA_MINUTOS, pat))
    return cands[0] if cands else None

# =========================
# KEY STATS PANEL
# =========================
def draw_key_stats_panel(home_name: str, away_name: str,
                         FERRO: Dict[str,float], RIVAL: Dict[str,float],
                         ferro_logo_path: Optional[str], rival_logo_path: Optional[str]):

    plt.close("all")
    fig_h = 0.66*len(ROW_ORDER) + 4.6
    fig = plt.figure(figsize=(10.8, fig_h))
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0), 1, 1, facecolor=bg_green, edgecolor="none", zorder=-10))

    # Banner sup
    BANNER_Y0 = 1.0 - (BANNER_H + 0.02)
    ax.add_patch(Rectangle((0, BANNER_Y0), 1, BANNER_H, facecolor="white", edgecolor="none", zorder=5))
    if ferro_logo_path: draw_logo(ax, ferro_logo_path, 0.09, BANNER_Y0 + BANNER_H*0.52, LOGO_W)
    if rival_logo_path: draw_logo(ax, rival_logo_path, 0.91, BANNER_Y0 + BANNER_H*0.52, LOGO_W)
    ax.text(0.5, BANNER_Y0 + BANNER_H*0.63, f"{home_name.upper()} vs {away_name.upper()}",
            ha="center", va="center", fontsize=TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, BANNER_Y0 + BANNER_H*0.29, "KEY STATS",
            ha="center", va="center", fontsize=SUB_FS, weight="bold", color=bg_green, zorder=7)

    # Banner inferior
    FOOTER_Y0 = 0.02
    ax.add_patch(Rectangle((0, FOOTER_Y0), 1, FOOTER_H, facecolor="white", edgecolor="none", zorder=5))
    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.63, "Trabajo Fin de Máster",
            ha="center", va="center", fontsize=FOOTER_TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.28, "Cristian Dieguez",
            ha="center", va="center", fontsize=FOOTER_SUB_FS,   weight="bold", color=bg_green, zorder=7)

    # Cuerpo
    EXTRA_GAP_BELOW_BANNER = 0.075
    top_y    = BANNER_Y0 - EXTRA_GAP_BELOW_BANNER
    bottom_y = FOOTER_Y0 + FOOTER_H + 0.012
    available_h = max(0.01, top_y - bottom_y)
    row_h = available_h / len(ROW_ORDER)

    mid_x, bar_w = 0.5, 0.33
    left_star_x, right_star_x = 0.055, 0.945

    def draw_row(y, label, lv, rv):
        label_y = y + row_h*(LABEL_Y_SHIFT_HIGH if RAISE_LABELS else LABEL_Y_SHIFT_LOW)
        ax.text(0.5, label_y, label, ha="center", va="center",
                fontsize=12.5, weight="bold", color=text_w)

        maxval = 100.0 if label in PERCENT_ROWS else max(1, lv, rv)
        lane_y = y + row_h*(0.22)
        lane_h = row_h*BAR_HEIGHT_FACTOR
        for x0 in (mid_x - bar_w - 0.02, mid_x + 0.02):
            ax.add_patch(Rectangle((x0, lane_y), bar_w, lane_h, facecolor=bar_rail, edgecolor=bar_rail))

        winner = "FERRO" if lv > rv else ("RIVAL" if rv > lv else "EMPATE")
        ferro_col   = orange_win if (USE_ORANGE_FOR_WIN and winner=="FERRO") else bar_white
        rival_col   = orange_win if (USE_ORANGE_FOR_WIN and winner=="RIVAL") else bar_rival
        ferro_alpha = 1.0 if winner!="RIVAL" else loser_alpha
        rival_alpha = 1.0 if winner!="FERRO" else loser_alpha

        scale = 0.88
        lw = 0 if maxval==0 else bar_w*(lv/maxval)*scale
        rw = 0 if maxval==0 else bar_w*(rv/maxval)*scale

        ltxt = f"{lv:.1f}%" if label in PERCENT_ROWS else f"{int(lv)}"
        rtxt = f"{rv:.1f}%" if label in PERCENT_ROWS else f"{int(rv)}"

        ax.add_patch(Rectangle((mid_x - 0.02 - lw, lane_y), lw, lane_h,
                               facecolor=ferro_col, edgecolor=ferro_col, alpha=ferro_alpha))
        ax.add_patch(Rectangle((mid_x + 0.02, lane_y), rw, lane_h,
                               facecolor=rival_col, edgecolor=rival_col, alpha=rival_alpha))
        ax.text(mid_x - bar_w - 0.030, lane_y + lane_h*0.45, ltxt, ha="right", va="center",
                fontsize=13, weight="bold", color=text_w)
        ax.text(mid_x + bar_w + 0.030, lane_y + lane_h*0.45, rtxt, ha="left",  va="center",
                fontsize=13, weight="bold", color=text_w)

        if winner == "FERRO":
            ax.text(left_star_x,  label_y, "★", ha="left",  va="center",
                    fontsize=14, color=star_c, weight="bold", clip_on=False)
        elif winner == "RIVAL":
            ax.text(right_star_x, label_y, "★", ha="right", va="center",
                    fontsize=14, color=star_c, weight="bold", clip_on=False)

    y = top_y
    for lab in ROW_ORDER:
        draw_row(y, lab, float(FERRO.get(lab,0)), float(RIVAL.get(lab,0)))
        y -= row_h

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================
# UI — MENÚ
# =========================
menu = st.sidebar.radio(
    "Menú",
    ["🏆 Tabla & Resultados","📊 Estadísticas de partido", "⏱️ Timeline de Partido", "🔥 Mapas de calor",
     "🕓 Distribución de minutos","🔗 Red de Pases", "📬 Destino de pases", 
     "🛡️ Pérdidas y Recuperaciones","🎯 Mapa de tiros", "📈 Radar comparativo"
    ],
    index=0
)


# =========================
# 📊 ESTADÍSTICAS DE PARTIDO
# =========================
if menu == "📊 Estadísticas de partido":
    matches_obj = discover_matches()
    matches = [m["label"] for m in matches_obj]
    if not matches:
        st.warning("No encontré partidos en data/minutos con patrón: 'Fecha N° - Rival - XML TotalValues.xml'.")
        st.stop()

    sel = st.selectbox("Elegí partido", matches, index=0)
    rival = rival_from_label(sel)

    # Después (para menús que usan XML de jugadores, p.ej. Mapa de tiros)
    match = get_match_by_label(sel)
    XML_PATH = match["xml_players"] if match else None
    # (si necesitás Matrix, podés seguir usando match["matrix_path"] si existe)
    MATRIX_PATH = match["matrix_path"] if match else None

    # 1) Posesión
    pos_m, pos_r = parse_possession_from_equipo(XML_PATH) if XML_PATH else (0.0, 0.0)

    # 2) Matrix
    if MATRIX_PATH and os.path.isfile(MATRIX_PATH):
        try:
            mx = compute_from_matrix(MATRIX_PATH)
            FERRO, RIVAL = mx["FERRO"], mx["RIVAL"]   # ✅ dentro del try
        except Exception as e:
            st.error(f"No pude leer MATRIX '{os.path.basename(MATRIX_PATH)}': {e}")
            FERRO, RIVAL = {}, {}                     # ✅ si falla, inicializa vacío
    else:
        FERRO, RIVAL = {}, {}

    # 3) Dibujar panel
    ferro_logo = badge_path_for("ferro")
    rival_logo = badge_path_for(rival)
    draw_key_stats_panel("Ferro", rival, FERRO, RIVAL, ferro_logo, rival_logo)


# =========================
# ⏱️ TIMELINE
# =========================
elif menu == "⏱️ Timeline de Partido":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos para timeline en data/minutos.")
        st.stop()

    sel = st.selectbox("Elegí partido", [m["label"] for m in matches], index=0)
    match = get_match_by_label(sel)
    if not match:
        st.error("No encontré info de XML para este partido.")
        st.stop()

    tl = build_timeline(match["xml_players"])
    ferro_logo = badge_path_for("ferro")
    rival_logo = badge_path_for(match["rival"])
    draw_timeline_panel(match["rival"], tl, ferro_logo, rival_logo)

# =========================
# 🔥 MAPAS DE CALOR
# =========================
elif menu == "🔥 Mapas de calor":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    sel = st.selectbox("Elegí partido", [m["label"] for m in matches], index=0)
    match = get_match_by_label(sel)
    df_raw = cargar_datos_nacsport(match["xml_players"]) if match else pd.DataFrame()
    df_xy = explode_coords_for_heatmap(df_raw)
    df_xy = rotate_coords_for_attack_right(df_xy)
    fig = fig_heatmap(df_xy, f"Mapa de calor {sel}")
    st.pyplot(fig, use_container_width=True)

# =========================
# 🕓 DISTRIBUCIÓN DE MINUTOS
# =========================
elif menu == "🕓 Distribución de minutos":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    sel = st.selectbox("Elegí partido", [m["label"] for m in matches], index=0)
    match = get_match_by_label(sel)
    df_pres = cargar_minutos_desde_xml_totalvalues(match["xml_players"]) if match else pd.DataFrame()
    df_por_rol, df_por_jugador = minutos_por_presencia(df_pres)

    st.subheader("⏱️ Minutos por jugador y rol")
    st.dataframe(df_por_rol)
    fig1 = fig_barh_minutos(df_por_rol["nombre"] + " (" + df_por_rol["rol"] + ")",
                            df_por_rol["segundos"], "Minutos por jugador y rol")
    st.pyplot(fig1, use_container_width=True)

    st.subheader("⏱️ Minutos totales por jugador")
    st.dataframe(df_por_jugador)
    fig2 = fig_barh_minutos(df_por_jugador["nombre"], df_por_jugador["segundos"], "Minutos totales por jugador")
    st.pyplot(fig2, use_container_width=True)

# =========================
# 🔗 RED DE PASES
# =========================
elif menu == "🔗 Red de Pases":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    sel = st.selectbox("Elegí partido", [m["label"] for m in matches], index=0)
    match = get_match_by_label(sel)
    df_raw = cargar_datos_nacsport(match["xml_players"]) if match else pd.DataFrame()
    fig = red_de_pases_por_rol(df_raw)
    st.pyplot(fig, use_container_width=True)

# =========================
# 🛡️ PÉRDIDAS Y RECUPERACIONES
# =========================
elif menu == "🛡️ Pérdidas y Recuperaciones":
    matches = discover_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos.")
        st.stop()

    sel = st.selectbox("Elegí partido", [m["label"] for m in matches], index=0)
    match = get_match_by_label(sel)
    df_raw = pr_cargar_datos(match["xml_players"]) if match else pd.DataFrame()
    df_pres = pr_cargar_presencias_equipo(match["xml_players"]) if match else pd.DataFrame()

    total_acc, perdidas, recupera, porc_perd, porc_recu, df_reg = pr_procesar(df_raw, df_pres, None)
    df_resumen = pr_resumen_df(total_acc, perdidas, recupera, porc_perd, porc_recu)

    st.subheader("📋 Resumen pérdidas/recuperaciones")
    st.dataframe(df_resumen)

    st.subheader("🔥 Heatmap de pérdidas (%)")
    st.pyplot(pr_heatmap(porc_perd, total_acc, "Pérdidas sobre total", good_high=False), use_container_width=True)

    st.subheader("🔥 Heatmap de recuperaciones (%)")
    st.pyplot(pr_heatmap(porc_recu, total_acc, "Recuperaciones sobre total", good_high=True), use_container_width=True)

    st.subheader("📊 Ranking de zonas con más pérdidas")
    st.pyplot(pr_bars(df_resumen, "%_perdidas_sobre_total", "Zonas con más pérdidas"), use_container_width=True)

    st.subheader("📊 Ranking de zonas con más recuperaciones")
    st.pyplot(pr_bars(df_resumen, "%_recuperaciones_sobre_total", "Zonas con más recuperaciones"), use_container_width=True)

# =========================
# 🎯 MAPA DE TIROS
# =========================
if menu == "🎯 Mapa de tiros":

    # ---- UI: Partido ----
    matches_obj = discover_matches()
    matches = [m["label"] for m in matches_obj]
    if not matches:
        st.warning("No encontré partidos en data/minutos con patrón: 'Fecha N° - Rival - XML TotalValues.xml'.")
        st.stop()

    sel = st.selectbox("Elegí partido", matches, index=0)
    rival = rival_from_label(sel)
    XML_PATH, _ = infer_paths_for_label(sel)

    if not XML_PATH or not os.path.isfile(XML_PATH):
        st.error("No encontré el XML de Jugadores/TotalValues para este partido.")
        st.stop()

    # ---- Helpers específicos del módulo de tiros (NO tocan nada del resto) ----
    def _is_shot_from_row(row) -> bool:
        # Dispara si en code/labels aparece "tiro" o "remate"
        code = nlower(row.get("jugador", ""))
        lbls = [nlower(l or "") for l in row.get("labels", [])]
        patt = re.compile(r"\b(tiro|remate)\b", re.I)
        if patt.search(code): return True
        return any(patt.search(l) for l in lbls)

    _ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")

    def _name_and_role(code: str) -> Tuple[str|None, str|None]:
        m = _ROLE_RE.match(code or "")
        if not m: return None, None
        return ntext(m.group(1)).strip(), ntext(m.group(2)).strip()

    # Clasificación estricta del resultado
    _KEYS = {
        "gol":        re.compile(r"\bgol\b", re.I),
        "ataj":       re.compile(r"\bataj", re.I),
        "al_arco":    re.compile(r"\btiro\s*al\s*arco\b", re.I),
        "bloqueado":  re.compile(r"\bbloquead", re.I),
        "desviado":   re.compile(r"\bdesviad", re.I),
        "errado":     re.compile(r"\berrad", re.I),
        "pifia":      re.compile(r"\bpifi", re.I),
    }
    def _shot_result_strict(code: str, labels: List[str]) -> str:
        s = nlower(code or ""); ll = [nlower(l or "") for l in (labels or [])]
        def _has(p): return bool(p.search(s) or any(p.search(x) for x in ll))
        if _has(_KEYS["gol"]):                         return "Gol"
        if _has(_KEYS["ataj"]) or _has(_KEYS["al_arco"]):  return "Tiro Atajado"
        if _has(_KEYS["bloqueado"]):                   return "Tiro Bloqueado"
        if _has(_KEYS["desviado"]):                    return "Tiro Desviado"
        if _has(_KEYS["errado"]) or _has(_KEYS["pifia"]):  return "Tiro Errado - Pifia"
        return "Sin clasificar"

    # Característica del origen
    _CHAR_PATTS = [
        ("de Corner (desde Banda)", re.compile(r"corner\s*\(desde\s*banda\)", re.I)),
        ("de Corner (centro)",      re.compile(r"corner\s*\(centro\)", re.I)),
        ("de Jugada (centro)",      re.compile(r"jugada\s*\(centro\)", re.I)),
        ("de Tiro Libre",           re.compile(r"tiro\s*libre", re.I)),
        ("de Rebote",               re.compile(r"\brebote\b", re.I)),
        ("de Lateral",              re.compile(r"\blateral\b", re.I)),
        ("de Jugada",               re.compile(r"\bde\s*jugada\b", re.I)),
    ]
    def _shot_char(code: str, labels: List[str]) -> str:
        s = nlower(code or "") + " " + " ".join(nlower(l or "") for l in (labels or []))
        for name, patt in _CHAR_PATTS:
            if patt.search(s): return name
        return "de Jugada"

    # Mapeo a cancha 35x20 (desde coords XML 0..20 x, 0..40 y)
    FLIP_TO_RIGHT = True
    FLIP_VERTICAL = True
    GOAL_PULL = 0.60  # "tirón" hacia el arco derecho

    def _map_raw_to_pitch(x_raw, y_raw, max_x, max_y, flip=True, pull=0.0, flip_v=False):
        x = (y_raw / max_y) * ANCHO
        y = (x_raw / max_x) * ALTO
        if flip:    x = ANCHO - x
        if flip_v:  y = ALTO - y
        if pull and 0.0 < pull < 1.0:
            x = x + pull * (ANCHO - x)
        x = float(np.clip(x, 0.0, ANCHO))
        y = float(np.clip(y, 0.0, ALTO))
        return x, y

    # ---- Cargar XML y preparar universe de filtros ----
    df_raw = cargar_datos_nacsport(XML_PATH)  # ya la tenés en tu app
    if df_raw.empty:
        st.info("Sin instancias en el XML.")
        st.stop()

    # Solo eventos de jugadores (evita códigos de equipo/posesión)
    df_raw = df_raw[df_raw["jugador"].apply(is_player_code)].copy()

    # Armar listas de jugadores y roles presentes solo en TIROS
    mask_shot = df_raw.apply(_is_shot_from_row, axis=1)
    df_shot = df_raw[mask_shot].copy()

    # Extraer nombre y rol desde "Nombre (Rol)"
    name_role = df_shot["jugador"].apply(_name_and_role)
    df_shot["player"] = name_role.apply(lambda t: t[0])
    df_shot["role"]   = name_role.apply(lambda t: t[1])

    players_present = sorted([p for p in df_shot["player"].dropna().unique()])
    roles_present   = sorted([r for r in df_shot["role"].dropna().unique()])

    # ---- UI: Filtros jugador/rol/característica ----
    sel_players = st.multiselect("Jugadores", players_present, default=players_present)
    sel_roles   = st.multiselect("Rol", roles_present, default=roles_present)
    char_opts   = ["Todas"] + [n for (n,_) in _CHAR_PATTS]
    sel_char    = st.selectbox("Característica del origen", char_opts, index=0)

    # ---- Construir lista de tiros (origen = punto más lejano al arco derecho) ----
    # Max X/Y reales del XML para escalar
    max_x = max((max((lst or [0])) for lst in df_shot["pos_x_list"]), default=19)
    max_y = max((max((lst or [0])) for lst in df_shot["pos_y_list"]), default=34)
    max_x = float(max_x if max_x else 19)
    max_y = float(max_y if max_y else 34)

    shots = []
    for _, r in df_shot.iterrows():
        player, role = r.get("player"), r.get("role")
        if player and sel_players and (player not in sel_players): continue
        if role and sel_roles and (role not in sel_roles): continue

        xs = r.get("pos_x_list") or []
        ys = r.get("pos_y_list") or []
        if not (xs and ys): continue

        # Mapeo de toda la trayectoria
        coords = [_map_raw_to_pitch(xr, yr, max_x, max_y,
                                    flip=FLIP_TO_RIGHT, pull=GOAL_PULL, flip_v=FLIP_VERTICAL)
                  for xr, yr in zip(xs, ys)]

        # Origen = más lejano al arco derecho → menor X después del flip
        origin = coords[0] if len(coords) == 1 else coords[int(np.argmin([c[0] for c in coords]))]

        res = _shot_result_strict(r.get("jugador",""), r.get("labels", []))
        ch  = _shot_char(r.get("jugador",""), r.get("labels", []))
        if sel_char != "Todas" and ch != sel_char:
            continue

        shots.append({"x": origin[0], "y": origin[1], "result": res, "char": ch,
                      "player": player, "role": role})

    # ---- Plot ----
    st.subheader("Mapa de tiros — Origen (punto más lejano al arco derecho)")
    plt.close("all")
    fig = plt.figure(figsize=(10.5, 7))
    ax  = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    draw_futsal_pitch_grid(ax)

    order = ["Gol","Tiro Atajado","Tiro Bloqueado","Tiro Desviado","Tiro Errado - Pifia","Sin clasificar"]
    COLORS = {
        "Gol":                "#FFD54F",
        "Tiro Atajado":       "#FFFFFF",
        "Tiro Bloqueado":     "#FF5252",
        "Tiro Desviado":      "#FF7043",
        "Tiro Errado - Pifia":"#6B6F76",
        "Sin clasificar":     "#BDBDBD",
    }

    for res in order:
        pts = [(s["x"], s["y"]) for s in shots if s["result"] == res]
        if not pts: continue
        xs, ys = zip(*pts)
        if res == "Gol":
            ax.scatter(xs, ys, s=160, c=COLORS[res], edgecolors="black", linewidths=0.6, zorder=5, label=res)
        elif res == "Tiro Atajado":
            ax.scatter(xs, ys, s=90,  c=COLORS[res], edgecolors="black", linewidths=0.6, zorder=4, label=res)
        elif res == "Tiro Bloqueado":
            ax.scatter(xs, ys, s=100, facecolors="none", edgecolors=COLORS[res], linewidths=1.8, zorder=4, label=res)
        elif res == "Tiro Desviado":
            ax.scatter(xs, ys, s=90,  facecolors="none", edgecolors=COLORS[res], linewidths=1.6, zorder=3, label=res)
        elif res == "Tiro Errado - Pifia":
            ax.scatter(xs, ys, s=110, marker='x', c=COLORS[res], linewidths=1.8, zorder=3, label=res)
        elif res == "Sin clasificar":
            ax.scatter(xs, ys, s=70,  c=COLORS[res], edgecolors="black", linewidths=0.4, zorder=2, label=res)

    f_players = ", ".join(sel_players) if sel_players else "Todos"
    f_roles   = ", ".join(sel_roles)   if sel_roles   else "Todos"
    f_char    = sel_char
    ax.set_title(
        f"{sel} — SHOTS (origen) | Jugadores: {f_players} | Roles: {f_roles} | Característica: {f_char}",
        fontsize=13, pad=6, weight="bold"
    )
    ax.legend(loc="upper left", frameon=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- Tablas (conteos y %) ----
    from collections import Counter
    res_order = ["Gol","Tiro Atajado","Tiro Bloqueado","Tiro Desviado","Tiro Errado - Pifia"]
    c_res = Counter(s["result"] for s in shots if s["result"] != "Sin clasificar")
    total_res = sum(c_res.values())
    df_res = pd.DataFrame({
        "Categoría": res_order + (["TOTAL"] if True else []),
        "Conteo": [c_res.get(k,0) for k in res_order] + [total_res],
        "%": [f"{(c_res.get(k,0)/total_res*100 if total_res else 0):.1f}%" for k in res_order] + ["100.0%" if total_res else "0.0%"]
    })

    c_char = Counter(s["char"] for s in shots)
    total_char = sum(c_char.values())
    rows_char = sorted(c_char.items(), key=lambda kv: (-kv[1], kv[0]))
    df_char = pd.DataFrame({
        "Categoría": [k for k,_ in rows_char] + ["TOTAL"],
        "Conteo": [v for _,v in rows_char] + [total_char],
        "%": [f"{(v/total_char*100 if total_char else 0):.1f}%" for _,v in rows_char] + ["100.0%" if total_char else "0.0%"]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Resultados del tiro**")
        st.dataframe(df_res, use_container_width=True)
    with col2:
        st.markdown("**Característica del origen**")
        st.dataframe(df_char, use_container_width=True)


# =========================
# 📬 DESTINO DE PASES (equipo / jugador / jugador-rol + tipos de pase)
# =========================
if menu == "📬 Destino de pases":
    # --- Partido ---
    matches = list_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos con patrón: 'Fecha N° - Rival - XML TotalValues.xml'.")
        st.stop()

    sel = st.selectbox("Elegí partido", matches, index=0)
    rival = rival_from_label(sel)

    # XML correcto: Jugadores (NacSport)
    mobj = get_match_by_label(sel)
    XML_PATH = None
    if mobj:
        XML_PATH = mobj.get("xml_players") or mobj.get("xml_jugadores") or mobj.get("xml_players_path")
    if not XML_PATH or not os.path.isfile(XML_PATH):
        XML_PATH, _ = infer_paths_for_label(sel)  # fallback TotalValues
        st.info("⚠️ No encontré 'XML NacSport' de jugadores; usando TotalValues como fallback.")
    if not XML_PATH or not os.path.isfile(XML_PATH):
        st.error("No encontré el XML del partido seleccionado.")
        st.stop()

    # --- Parámetros de dibujo ---
    ANCHO, ALTO = 35.0, 20.0
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        myteam_right = st.checkbox("Mi equipo ataca → (derecha)", value=True)
    with colB:
        show_rival = st.checkbox("Mostrar pases del rival", value=False)
    with colC:
        show_last_third = st.checkbox("Resaltar destino en último tercio", value=True)

    # --- Alcance / filtros ---
    scope = st.radio("Ámbito", ["Equipo", "Por jugador", "Por jugador (rol)"], horizontal=True)

    # === Parser correcto para este menú (xs/ys + labels) ===
    import xml.etree.ElementTree as ET
    def parse_events_with_coords(xml_path):
        """Devuelve lista de eventos con: code, labels(list lower), xs(list int), ys(list int)."""
        out = []
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as e:
            st.error(f"No pude leer XML: {e}")
            return out
        for inst in root.findall(".//instance"):
            code = ntext(inst.findtext("code"))
            labels = [nlower(t.text) for t in inst.findall("./label/text")]
            xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
            ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
            out.append({"code": code, "labels": labels, "xs": xs, "ys": ys})
        return out

    # --- Armado de listas de jugadores/roles desde el XML ---
    events = parse_events_with_coords(XML_PATH)  # <<<< AQUÍ va el reemplazo del "ANTES/DESPUÉS"
    jugadores_set, roles_set = set(), set()
    for e in events:
        code = e.get("code") or ""
        if is_player_code(code):
            m = _NAME_ROLE_RE.match(code)
            if m:
                jugadores_set.add(ntext(m.group(1)).strip())
                roles_set.add(ntext(m.group(2)).strip())
    jugadores = sorted(jugadores_set)
    roles     = sorted(r for r in roles_set if r)

    sel_players = []
    sel_roles   = []
    if scope != "Equipo":
        colP, colR = st.columns(2)
        with colP:
            sel_players = st.multiselect("Jugador(es)", jugadores, default=[])
        with colR:
            if scope == "Por jugador (rol)":
                sel_roles = st.multiselect("Rol(es)", roles, default=[])

    # --- Tipos de pase ---
    PASS_TYPES = {
        "Asistencia":        [r"\b(asist|asistencia|assist|pase\s*de?\s*gol)\b"],
        "Pase clave":        [r"\b(pase\s*clave|key\s*pass|keypass)\b"],
        "Progresivo frontal":[r"\bprogresivo\b.*\bfrontal\b"],
        "Progresivo lateral":[r"\bprogresivo\b.*\blateral\b"],
        "Corto frontal":     [r"\bcorto\b.*\bfrontal\b"],
        "Corto lateral":     [r"\bcorto\b.*\blateral\b"],
        "Largo frontal":     [r"\blargo\b.*\bfrontal\b"],
        "Largo lateral":     [r"\blargo\b.*\blateral\b"],
    }
    tipos_opciones = list(PASS_TYPES.keys())
    selected_types = st.multiselect("Tipos de pase (vacío = todos)", tipos_opciones, default=[])

    # --- Detectores / helpers ---
    import re
    ASSIST_PAT  = re.compile(r"\b(asist|asistencia|assist|pase\s*de?\s*gol)\b", re.I)
    KEYPASS_PAT = re.compile(r"\b(pase\s*clave|key\s*pass|keypass)\b", re.I)

    def is_pass_attempt(ev) -> bool:
        s = nlower(ev.get("code",""))
        if re.match(r"^\s*pase\b", s): return True
        return any(re.match(r"^\s*pase\b", nlower(l or "")) for l in ev.get("labels", []))

    def is_rival_evt(ev) -> bool:
        return nlower(ev.get("code","")).startswith("categoria - equipo rival")

    def match_pass_type(ev) -> set:
        txt = nlower(ev.get("code","")) + " " + " ".join([nlower(l or "") for l in ev.get("labels",[])])
        found = set()
        if ASSIST_PAT.search(txt):  found.add("Asistencia")
        if KEYPASS_PAT.search(txt): found.add("Pase clave")
        for name, patt_list in PASS_TYPES.items():
            if name in {"Asistencia","Pase clave"}:
                continue
            for p in patt_list:
                if re.search(p, txt):
                    found.add(name); break
        return found

    def map_raw_to_pitch(x_raw, y_raw):
        if x_raw is None or y_raw is None:
            return (None, None)
        X = 35.0 - (float(y_raw) * (35.0 / 40.0))  # y→X, invierte eje
        Y = float(x_raw)                            # x→Y
        if not myteam_right:
            X = 35.0 - X
        return (max(0.0, min(35.0, X)), max(0.0, min(20.0, Y)))

    def closer_to_goal(ax, bx):
        return (bx > ax) if myteam_right else (bx < ax)

    # --- Recolección / filtrado ---
    from matplotlib.collections import LineCollection
    segs_yellow, segs_orange, segs_grey = [], [], []
    dest_assist, dest_key, dest_last, dest_other = [], [], [], []
    kept = 0

    for ev in events:
        if not is_pass_attempt(ev):
            continue

        # Equipo (mi equipo / rival)
        if not show_rival and is_rival_evt(ev):
            continue
        if show_rival and not is_rival_evt(ev):
            continue

        # Jugador / Rol
        code = ev.get("code","")
        m = _NAME_ROLE_RE.match(code) if code else None
        nombre = ntext(m.group(1)).strip() if m else None
        rol    = ntext(m.group(2)).strip() if m else None   # <<<< AQUÍ va lo de nombre/rol

        if scope != "Equipo":
            if sel_players and (nombre not in sel_players):  continue
            if scope == "Por jugador (rol)" and sel_roles and (rol not in sel_roles): continue

        # Tipos de pase
        types_found = match_pass_type(ev)
        if selected_types and len(types_found.intersection(selected_types)) == 0:
            continue

        xs = ev.get("xs") or []; ys = ev.get("ys") or []
        if not (xs and ys):  continue

        x0, y0 = map_raw_to_pitch(xs[0],  ys[0])
        x1, y1 = map_raw_to_pitch(xs[-1], ys[-1])
        if None in (x0,y0,x1,y1): continue

        # Elegir destino más cercano al arco rival (por X)
        if closer_to_goal(x0, x1):
            start_plot = (x0,y0); end_plot = (x1,y1)
        elif closer_to_goal(x1, x0):
            start_plot = (x1,y1); end_plot = (x0,y0)
        else:
            start_plot = (x0,y0); end_plot = (x1,y1)

        kept += 1

        if "Asistencia" in types_found:
            segs_yellow.append([start_plot, end_plot]); dest_assist.append(end_plot)
        elif "Pase clave" in types_found:
            segs_orange.append([start_plot, end_plot]); dest_key.append(end_plot)
        else:
            segs_grey.append([start_plot, end_plot])
            if show_last_third:
                in_last = (end_plot[0] >= (2/3)*ANCHO) if myteam_right else (end_plot[0] <= (1/3)*ANCHO)
                (dest_last if in_last else dest_other).append(end_plot)
            else:
                dest_other.append(end_plot)

    # --- Plot ---
    plt.close("all")
    fig = plt.figure(figsize=(10.6, 7.0))
    ax  = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    draw_futsal_pitch_grid(ax)

    if segs_grey:
        ax.add_collection(LineCollection(segs_grey, colors="#9BA3AE", linewidths=1.4, alpha=0.65, zorder=3))
    if segs_orange:
        ax.add_collection(LineCollection(segs_orange, colors="#FFA726", linewidths=2.0, alpha=0.95, zorder=4))
    if segs_yellow:
        ax.add_collection(LineCollection(segs_yellow, colors="#FFB300", linewidths=2.4, alpha=0.95, zorder=5))

    if dest_other:
        xs, ys = zip(*dest_other)
        ax.scatter(xs, ys, s=42, facecolors="#A0A7B2", edgecolors="black", linewidths=0.2, zorder=4, label="Destino (otros)")
    if dest_last:
        xs, ys = zip(*dest_last)
        ax.scatter(xs, ys, s=52, facecolors="#2E86FF", edgecolors="black", linewidths=0.3, zorder=5, label="Destino (último tercio)")
    if dest_key:
        xs, ys = zip(*dest_key)
        ax.scatter(xs, ys, s=64, facecolors="#FFD54F", edgecolors="black", linewidths=0.4, zorder=6, label="Pase clave (destino)")
    if dest_assist:
        xs, ys = zip(*dest_assist)
        ax.scatter(xs, ys, marker='*', s=220, c="#FFEB3B", edgecolors="none", zorder=7, label="Asistencia (destino)")

    who = "Equipo" if scope=="Equipo" else (
        f"Jugadores: {', '.join(sel_players) if sel_players else 'Todos'}"
        + (f" | Roles: {', '.join(sel_roles) if sel_roles else 'Todos'}" if scope=="Por jugador (rol)" else "")
    )
    ttypes = "Todos" if not selected_types else ", ".join(selected_types)
    side = "→" if myteam_right else "←"
    ax.set_title(f"Destino de pases — {who}  |  Tipos: {ttypes}  |  Ataque {side}", fontsize=13, pad=6, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(f"Pases plotteados: {kept}")


# =========================
# 📈 RADAR COMPARATIVO — Jugador total / Por rol / Jugador & Rol
# Cálculo on-the-fly: Minutos (XML TotalValues) + Matrix (XLSX)
# =========================
if menu == "📈 Radar comparativo":
    import os, re, unicodedata as ud, xml.etree.ElementTree as ET
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pi

    # ---------- UI: rutas base ----------
    st.subheader("Radar comparativo (multijugador / multirol)")
    colr1, colr2 = st.columns(2)
    with colr1:
        dir_minutos = st.text_input("Carpeta de MINUTOS (XML TotalValues)", value="data/minutos")
    with colr2:
        dir_matrix  = st.text_input("Carpeta de MATRIX (XLSX)", value="data/matrix")

    DIR_MINUTOS = Path(dir_minutos).expanduser().resolve()
    DIR_MATRIX  = Path(dir_matrix).expanduser().resolve()
    if not DIR_MINUTOS.exists():
        st.error(f"No existe carpeta MINUTOS: {DIR_MINUTOS}")
        st.stop()
    if not DIR_MATRIX.exists():
        st.error(f"No existe carpeta MATRIX: {DIR_MATRIX}")
        st.stop()

    # ---------- helpers texto ----------
    def norm_txt(s: str) -> str:
        if s is None: return ""
        s = str(s)
        s = ud.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
        s = re.sub(r"\s+", " ", s.strip().lower())
        return s

    NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)")
    RE_FECHA_RIVAL_MIN = re.compile(
        r"Fecha\s*(?:N[º°o]\.?\s*)?(\d+)\s*-\s*(.+?)\s*-\s*XML\s*TotalValues",
        re.IGNORECASE
    )
    RE_FECHA_RIVAL_MAT = re.compile(
        r"Fecha\s*(\d+)\s*-\s*(.+?)\s*-\s*Matrix",
        re.IGNORECASE
    )

    def parse_fecha_rival_from_name(stem: str, is_minutos=True):
        m = (RE_FECHA_RIVAL_MIN if is_minutos else RE_FECHA_RIVAL_MAT).search(stem)
        if m:
            fecha = int(m.group(1))
            rival = m.group(2).strip()
            return fecha, rival, f"F{fecha}_{rival}"
        # fallback
        return None, stem, stem

    def mmss(seg):
        s = int(round(seg))
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"

    # ---------- 1) MINUTOS desde XML TotalValues ----------
    @st.cache_data(show_spinner=True)
    def build_minutos_por_partido(DIR_MINUTOS: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Devuelve:
          - df_por_rol: minutos/partido por (nombre, rol)
          - df_por_jugador: minutos/partido total por jugador (sumando roles)
        """
        HARD_LABELS = {
            "goles a favor en cancha": "gf_on",
            "gol rival en cancha": "ga_on",
            "participa en gol hecho": "part_gf",
            "involucrado en gol recibido": "invol_ga",
            "valla invicta en cancha": "valla",
        }
        IGNORABLE_LABELS = {"total"}

        def labels_from_inst(inst):
            labs = []
            for lab in inst.findall("./label"):
                txt = lab.findtext("text")
                if isinstance(txt, str) and txt.strip():
                    labs.append(norm_txt(txt))
            labs = [x for x in labs if x]
            decision = [x for x in labs if x not in IGNORABLE_LABELS]
            return labs, decision

        def merge_intervals(intervals):
            if not intervals: return []
            ints = []
            for s, e in intervals:
                try:
                    s = float(s); e = float(e)
                except:
                    continue
                if e <= s: e = s + 0.04
                ints.append((s, e))
            ints.sort()
            merged = [list(ints[0])]
            for s, e in ints[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            return [(s, e) for s, e in merged]

        rows_rol, rows_jug = [], []
        xmls = sorted(DIR_MINUTOS.rglob("*.xml"))
        for p in xmls:
            if "TotalValues" not in p.stem:
                continue
            fecha, rival, partido_id = parse_fecha_rival_from_name(p.stem, is_minutos=True)
            try:
                root = ET.parse(p).getroot()
            except Exception as e:
                continue

            agg = {}  # (nombre, rol) -> {"intervals":[(s,e)], contadores...}
            for inst in root.findall(".//instance"):
                s = inst.findtext("start"); e = inst.findtext("end")
                try:
                    s = float(s); e = float(e)
                except:
                    continue
                if e <= s: e = s + 0.04

                code = inst.findtext("code") or ""
                m = NAME_ROLE_RE.match(code)
                if not m:  # asegurar solo instancias con "Nombre (Rol)"
                    continue
                nombre = m.group(1).strip()
                rol    = m.group(2).strip()

                labs_all, labs_dec = labels_from_inst(inst)
                has_relevant = any(l in HARD_LABELS for l in labs_all)
                is_empty_after = (len(labs_dec) == 0)
                if not (has_relevant or is_empty_after):
                    continue

                counts = {k:0 for k in ("gf_on","ga_on","part_gf","invol_ga","valla")}
                for l in labs_all:
                    if l in HARD_LABELS:
                        counts[HARD_LABELS[l]] += 1

                key = (nombre, rol)
                if key not in agg:
                    agg[key] = {"intervals": [], "gf":0, "ga":0, "part":0, "invol":0, "valla":0}
                a = agg[key]
                a["intervals"].append((s, e))
                a["gf"]    += counts["gf_on"]
                a["ga"]    += counts["ga_on"]
                a["part"]  += counts["part_gf"]
                a["invol"] += counts["invol_ga"]
                a["valla"] += counts["valla"]

            # construir filas
            for (nombre, rol), a in agg.items():
                merged = merge_intervals(a["intervals"])
                segundos = sum(e - s for s, e in merged)
                rows_rol.append({
                    "partido_id": partido_id, "fecha": fecha, "rival": rival,
                    "nombre": nombre, "rol": rol,
                    "segundos": int(round(segundos)),
                    "mmss": mmss(segundos),
                    "minutos": round(segundos/60.0, 2),
                })

            # jugador total
            if agg:
                by_player = {}
                for (nombre, rol), a in agg.items():
                    merged = merge_intervals(a["intervals"])
                    seg = sum(e - s for s, e in merged)
                    by_player[nombre] = by_player.get(nombre, 0.0) + seg
                for nombre, seg in by_player.items():
                    rows_jug.append({
                        "partido_id": partido_id, "fecha": fecha, "rival": rival,
                        "nombre": nombre,
                        "segundos": int(round(seg)),
                        "mmss": mmss(seg),
                        "minutos": round(seg/60.0, 2),
                    })

        df_por_rol = pd.DataFrame(rows_rol)
        df_por_jug = pd.DataFrame(rows_jug)
        return df_por_rol, df_por_jug

    # ---------- 2) MATRIX desde XLSX (cuenta acciones + % derivados) ----------
    GROUPS = {
        "Pase Corto Frontal": ["Pase Corto Frontal", "Pase Corto Frontal Completado", "Pase Corto Frontal OK"],
        "Pase Corto Lateral": ["Pase Corto Lateral", "Pase Corto Lateral Completado", "Pase Corto Lateral OK"],
        "Pase Progresivo Frontal": ["Pase Progresivo Frontal", "Pase Progresivo Frontal Completado", "Pase Progresivo Frontal OK"],
        "Pase Progresivo Lateral": ["Pase Progresivo Lateral", "Pase Progesivo Lateral Completado", "Pase Progresivo Lateral OK"],
        "Centros": ["Centros", "Centros OK", "Centros Rematados"],
        "Tiros": ["Tiro al arco","Tiro Atajado","Tiro Bloqueado","Tiro Desviado","Tiro Errado - Pifia","Tiro Hecho"],
        "Regates": [
            "Regate conseguido - Mantiene pelota",
            "Regate conseguido - Pierde pelota",
            "Regate No conseguido - Mantiene pelota",
            "Regate No conseguido - Pierde pelota",
        ],
        "Pivot": ["Aguanta Pivotea","Gira"],
        "Presion": ["Presiona","Presionado"],
        "Faltas": ["Faltas Hecha","Faltas Recibidas"],
        "Recuperaciones": ["Recuperacion x Duelo","Recuperación x Interceptacion","Recuperacion x Mal Control",
                           "Recuperacion x Mal Pase Rival","Recuperacion x Robo"],
        "Perdidas": ["Pérdida x Duelo","Pérdida x Interceptacion Rival","Pérdida x Mal Control","Pérdida x Mal Pase","Pérdida x Robo Rival"],
        "1v1": ["1v1 Ganado","1v1 perdido"],
        "Arquero Acciones": ["Accion fuera del area","Achique","Rebote corto","Rebote largo","Tiro Atajado por Reflejos"],
        "Arquero Pase Corto Frontal": ["Pase Corto Frontal cPie","Pase Corto Frontal Completado cPie","Pase Corto Frontal OK cPie"],
        "Arquero Pase Corto Lateral": ["Pase Corto Lateral cPie","Pase Corto Lateral Completado cPie","Pase Corto Lateral OK cPie"],
        "Arquero Pase Progresivo Frontal": ["Pase Progresivo Frontal cPie","Pase Progesivo Frontal Completado cPie","Pase Progresivo Frontal OK cPie"],
        "Arquero Pase Progresivo Lateral": ["Pase Progresivo Lateral cPie","Pase Progesivo Lateral Completado cPie","Pase Progresivo Lateral OK cPie"],
        "Arquero Salida Corta": ["Salida de arco corto cMano","Salida de arco Corto Completado cMano","Salida de arco Corto OK cMano"],
        "Arquero Salida Progresiva": ["Salida de arco progresivo cMano","Salida de arco Progesivo Completado cMano","Salida de arco Progresivo OK cMano"],
        "Corners": ["Corner 2do Palo","Corner Al Area","Corner En corto"],
        "Asistencias/Clave": ["Asistencia","Pase Clave"],
        "Conduccion": ["Conduccion"],
        "Despeje": ["Despeje"],
        "Gol": ["Gol"],
    }
    NORM_TO_ORIG = {norm_txt(v): v for vs in GROUPS.values() for v in vs}
    ALL_ACTIONS = list(dict.fromkeys([v for vs in GROUPS.values() for v in vs]))

    def split_name_role(code: str):
        m = NAME_ROLE_RE.match(str(code))
        if m: return m.group(1).strip(), m.group(2).strip()
        return None, None

    def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        if not df.columns.duplicated().any():
            return df
        merged = {}
        for col in df.columns.unique():
            block = df.loc[:, df.columns == col]
            if block.shape[1] == 1:
                merged[col] = block.iloc[:, 0]
            else:
                num = block.apply(pd.to_numeric, errors="coerce")
                summed = num.sum(axis=1, min_count=1)
                if summed.notna().any():
                    merged[col] = summed
                else:
                    merged[col] = block.bfill(axis=1).iloc[:, 0]
        return pd.DataFrame(merged)

    def normalize_columns(df):
        df = df.copy()
        df.columns = [norm_txt(c) for c in df.columns]
        return df

    def wide_or_long_to_wide(df):
        df = df.copy()
        cols = set(df.columns)
        # reconstrucción jugador/rol
        if "code" in cols and ("jugador" not in cols or "rol" not in cols):
            j, r = zip(*df["code"].map(split_name_role))
            df["jugador"] = list(j); df["rol"] = list(r)

        if "jugador" in df.columns and "rol" not in df.columns:
            j, r = zip(*df["jugador"].map(split_name_role))
            df["_j"] = j; df["_r"] = r
            df["rol"] = df["_r"].where(pd.Series(r).notna(), df.get("rol","sin rol"))
            df["jugador"] = df["_j"].where(pd.Series(j).notna(), df["jugador"])
            df = df.drop(columns=["_j","_r"])

        if "jugador" not in df.columns or "rol" not in df.columns:
            if "nombre" in df.columns and "rol" in df.columns:
                df = df.rename(columns={"nombre":"jugador"})
            else:
                return pd.DataFrame(columns=["jugador","rol","accion","cantidad"])

        df = df[(df["jugador"].astype(str).str.strip().str.lower() != "tiempo no jugado")]
        # ¿long ya?
        var_col = next((c for c in ("variable","accion","accion/variable","evento") if c in cols), None)
        val_col = next((c for c in ("cantidad","valor","count","conteo","n") if c in cols), None)
        if var_col and val_col:
            tmp = df.loc[:, ["jugador","rol",var_col,val_col]].copy()
            tmp = tmp.rename(columns={var_col:"accion", val_col:"cantidad"})
            tmp["accion"] = tmp["accion"].map(lambda x: NORM_TO_ORIG.get(norm_txt(x), x))
            tmp["cantidad"] = pd.to_numeric(tmp["cantidad"], errors="coerce").fillna(0)
            tmp = tmp.groupby(["jugador","rol","accion"], as_index=False)["cantidad"].sum()
            return tmp

        # wide → long
        action_cols = []
        for c in df.columns:
            if c in ("jugador","rol","code"): 
                continue
            col_obj = pd.to_numeric(df[c], errors="coerce")
            if col_obj.notna().any() or norm_txt(c) in NORM_TO_ORIG:
                action_cols.append(c)
        if not action_cols:
            return pd.DataFrame(columns=["jugador","rol","accion","cantidad"])

        long = df[["jugador","rol"] + action_cols].melt(id_vars=["jugador","rol"], var_name="accion", value_name="cantidad")
        long["cantidad"] = pd.to_numeric(long["cantidad"], errors="coerce").fillna(0)
        long["accion"] = long["accion"].map(lambda x: NORM_TO_ORIG.get(norm_txt(x), x))
        long = long.groupby(["jugador","rol","accion"], as_index=False)["cantidad"].sum()
        return long

    def counts_per_jugrol(long_df):
        if long_df.empty: return pd.DataFrame()
        rows = []
        for (jug, rol), g in long_df.groupby(["jugador","rol"]):
            row = {"jugador": jug, "rol": rol}
            dd = dict(g.groupby("accion")["cantidad"].sum())
            for a in ALL_ACTIONS:
                row[a] = dd.get(a, 0)
            rows.append(row)
        return pd.DataFrame(rows)

    def ratio(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / den, np.nan)
        return r

    def add_derived_percentages(df):
        if df.empty: return df
        df = df.copy()

        def s(*cols):
            vals = [pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0) for c in cols if c in df.columns]
            return sum(vals) if vals else pd.Series(0, index=df.index)

        # Regates
        rcols = GROUPS["Regates"]
        if all(c in df.columns for c in rcols):
            df["Regates - Total"] = s(*rcols)
            exi = pd.to_numeric(df["Regate conseguido - Mantiene pelota"], errors="coerce").fillna(0)
            df["% Regates Exitosos"] = ratio(exi, df["Regates - Total"]) * 100

        # 1v1
        if "1v1 Ganado" in df.columns and "1v1 perdido" in df.columns:
            g1 = pd.to_numeric(df["1v1 Ganado"], errors="coerce").fillna(0)
            p1 = pd.to_numeric(df["1v1 perdido"], errors="coerce").fillna(0)
            df["% Duelos Ganados"] = ratio(g1, g1 + p1) * 100

        # Tiros
        if "Tiro al arco" in df.columns and "Tiro Hecho" in df.columns:
            ta = pd.to_numeric(df["Tiro al arco"], errors="coerce").fillna(0)
            th = pd.to_numeric(df["Tiro Hecho"], errors="coerce").fillna(0)
            gol = pd.to_numeric(df.get("Gol", 0), errors="coerce").fillna(0)
            df["Tiros - % al arco"] = ratio(ta, th) * 100
            df["Tiros - % Goles/Tiro al arco"] = ratio(gol, ta) * 100

        # Recuperaciones / Pérdidas
        rec_cols = GROUPS["Recuperaciones"]
        per_cols = GROUPS["Perdidas"]
        df["Recuperaciones - Total"] = s(*rec_cols)
        df["Perdidas - Total"]      = s(*per_cols)
        df["% Recuperaciones"]      = ratio(df["Recuperaciones - Total"], df["Recuperaciones - Total"] + df["Perdidas - Total"]) * 100

        # % Pases (OK/Base y Completado/Base)
        passlike = [k for k in GROUPS if k.startswith("Pase ") or k.startswith("Arquero Pase")]
        for gname in passlike:
            base, comp, ok = GROUPS[gname]
            if base in df.columns:
                b = pd.to_numeric(df[base], errors="coerce").fillna(0)
                c = pd.to_numeric(df.get(comp, 0), errors="coerce").fillna(0)
                o = pd.to_numeric(df.get(ok, 0), errors="coerce").fillna(0)
                df[f"% {base}"] = ratio(o, b) * 100
                df[f"% {comp}"] = ratio(c, b) * 100

        # Índice positivo (simple)
        posit = []
        # “Completados”
        posit += [c for c in df.columns if "Completado" in c and not c.strip().startswith("%")]
        posit += ["Centros Rematados","Tiro al arco",
                  "Regate conseguido - Mantiene pelota",
                  "Aguanta Pivotea","Gira","Faltas Recibidas",
                  "Recuperaciones - Total","1v1 Ganado","Asistencia","Pase Clave","Gol","Conduccion"]
        posit = list(dict.fromkeys([c for c in posit if c in df.columns]))
        tot = [c for c in df.columns if c in ALL_ACTIONS]  # todas las acciones base presentes

        def s_cols(cols): 
            return sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in cols) if cols else pd.Series(0, index=df.index)

        df["Acciones Positivas - Total"] = s_cols(posit)
        df["Acciones - Total"] = s_cols(tot)
        df["% Acciones Positivas"] = ratio(df["Acciones Positivas - Total"], df["Acciones - Total"]) * 100
        return df

    @st.cache_data(show_spinner=True)
    def build_matrix_counts(DIR_MATRIX: Path) -> pd.DataFrame:
        rows = []
        xlsx_files = sorted(DIR_MATRIX.glob("*.xlsx"))
        for xlsx in xlsx_files:
            fecha, rival, partido_id = parse_fecha_rival_from_name(xlsx.stem, is_minutos=False)
            try:
                sheets = pd.read_excel(xlsx, sheet_name=None)
            except Exception:
                continue
            for sh, df in (sheets or {}).items():
                if df is None or df.empty: 
                    continue
                df = normalize_columns(df)
                df = merge_duplicate_columns(df)

                # eliminar columnas alineación tipo "Nombre (rol)" si aparecen como extra
                alineacion_pat = re.compile(r".+\((?:cierre|pivot|ala\s*[di])\)$", flags=re.I)
                cols_alineacion = [c for c in df.columns if alineacion_pat.match(str(c))]
                if cols_alineacion:
                    df = df.drop(columns=cols_alineacion, errors="ignore")

                long_df = wide_or_long_to_wide(df)
                if long_df.empty: 
                    continue
                cnt = counts_per_jugrol(long_df)
                if cnt.empty: 
                    continue
                cnt.insert(0, "partido_id", partido_id)
                cnt.insert(1, "fecha", fecha)
                cnt.insert(2, "rival", rival)
                cnt.insert(3, "archivo", xlsx.name)
                cnt.insert(4, "hoja", sh)
                rows.append(cnt)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if out.empty:
            return out
        out = add_derived_percentages(out)
        return out

    # ---------- Construir datasets base (caché) ----------
    with st.spinner("Leyendo MINUTOS y MATRIX..."):
        df_min_rol, df_min_jug = build_minutos_por_partido(DIR_MINUTOS)
        df_mat_rol = build_matrix_counts(DIR_MATRIX)

    if df_mat_rol.empty:
        st.warning("No se pudo leer MATRIX (XLSX). Revisa la carpeta.")
        st.stop()
    if df_min_rol.empty and df_min_jug.empty:
        st.warning("No se pudo leer MINUTOS (XML TotalValues). Revisa la carpeta.")
        st.stop()

    # ---------- Intersección de partidos válidos & normalización a 40' ----------
    # Por ROL
    if not df_min_rol.empty:
        join_rol = pd.merge(
            df_mat_rol,
            df_min_rol[["partido_id","fecha","rival","nombre","rol","minutos"]],
            left_on=["partido_id","fecha","rival","jugador","rol"],
            right_on=["partido_id","fecha","rival","nombre","rol"],
            how="inner"
        ).drop(columns=["nombre"])
    else:
        join_rol = pd.DataFrame()

    # Por JUGADOR total (sumando roles)
    if not df_min_jug.empty:
        join_jug = pd.merge(
            df_mat_rol,
            df_min_jug[["partido_id","fecha","rival","nombre","minutos"]],
            left_on=["partido_id","fecha","rival","jugador"],
            right_on=["partido_id","fecha","rival","nombre"],
            how="inner"
        ).drop(columns=["nombre"])
    else:
        join_jug = pd.DataFrame()

    # Columnas % y absolutas
    def split_pct_abs(df):
        pct_cols = [c for c in df.columns if "%" in c]
        id_cols  = ["partido_id","fecha","rival","archivo","hoja","jugador","rol","minutos"]
        abs_cols = [c for c in df.columns if c not in pct_cols + id_cols]
        return abs_cols, pct_cols

    # Normalizar ABS por partido a 40 min, luego promediar
    def normalize_and_avg(df, by_cols):
        if df.empty: 
            return df
        abs_cols, pct_cols = split_pct_abs(df)
        out = df.copy()
        for c in abs_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c] * (40.0 / out["minutos"].replace(0, np.nan))
        # promedios por entidad
        grouped = out.groupby(by_cols, as_index=False)[abs_cols + pct_cols].mean(numeric_only=True)
        # redondeos amables
        grouped[abs_cols] = grouped[abs_cols].round(2)
        grouped[pct_cols] = grouped[pct_cols].round(2)
        return grouped

    df_avg_por_rol      = normalize_and_avg(join_rol, by_cols=["jugador","rol"])
    df_avg_por_jugador  = normalize_and_avg(join_jug, by_cols=["jugador"])

    # ---------- UI de comparación ----------
    scope = st.radio("Ámbito de comparación", ["Jugador total", "Por rol", "Jugador y rol"], horizontal=True)

    # lista de métricas elegibles (todas las numéricas disponibles)
    candidates = sorted([c for c in df_mat_rol.columns if c not in 
                        {"partido_id","fecha","rival","archivo","hoja","jugador","rol"}])
    # separamos por tipo
    pct_candidates = [c for c in candidates if "%" in c]
    abs_candidates = [c for c in candidates if "%" not in c]

    st.markdown("**Elegí métricas (pueden ser % y/o absolutos normalizados a 40’)**")
    c1, c2 = st.columns(2)
    with c1:
        mets_pct = st.multiselect("Métricas %", pct_candidates, default=["% Acciones Positivas"])
    with c2:
        mets_abs = st.multiselect("Métricas absolutas", abs_candidates, default=[])

    metrics = [*mets_pct, *mets_abs]
    if not metrics:
        st.info("Elegí al menos una métrica para comparar.")
        st.stop()

    min_minutos = st.number_input("Minutos mínimos totales (para incluir en la comparación)", min_value=0.0, value=20.0, step=5.0)

    # filtros según scope
    if scope == "Jugador total":
        base = df_avg_por_jugador.copy()
        if base.empty:
            st.warning("No hay datos válidos para Jugador total.")
            st.stop()
        # sumar minutos válidos
        mins_valid = join_jug.groupby("jugador", as_index=False)["minutos"].sum()
        base = pd.merge(base, mins_valid, on="jugador", how="left", suffixes=("","_tot"))
        base = base[base["minutos"] >= min_minutos].copy()
        # selección de jugadores
        jugadores = sorted(base["jugador"].unique().tolist())
        sel_jug = st.multiselect("Jugadores a comparar", jugadores, default=jugadores[:min(6, len(jugadores))])
        base = base[base["jugador"].isin(sel_jug)]
        label_col = "jugador"

    elif scope == "Por rol":
        base = df_avg_por_rol.copy()
        if base.empty:
            st.warning("No hay datos válidos para Rol.")
            st.stop()
        # minutos por jug+rol
        mins_valid = join_rol.groupby(["jugador","rol"], as_index=False)["minutos"].sum()
        base = pd.merge(base, mins_valid, on=["jugador","rol"], how="left")
        # filtro rol
        roles = sorted(base["rol"].dropna().unique().tolist())
        sel_rol = st.selectbox("Rol", roles, index=0 if roles else None)
        base = base[base["rol"] == sel_rol]
        base = base[base["minutos"] >= min_minutos].copy()
        jugadores = sorted(base["jugador"].unique().tolist())
        sel_jug = st.multiselect("Jugadores a comparar", jugadores, default=jugadores[:min(6, len(jugadores))])
        base = base[base["jugador"].isin(sel_jug)]
        label_col = "jugador"

    else:  # Jugador y rol (comparar entradas jug-rol entre sí)
        base = df_avg_por_rol.copy()
        if base.empty:
            st.warning("No hay datos válidos para Jugador & Rol.")
            st.stop()
        mins_valid = join_rol.groupby(["jugador","rol"], as_index=False)["minutos"].sum()
        base = pd.merge(base, mins_valid, on=["jugador","rol"], how="left")
        # escoger combinaciones
        base["jug_rol"] = base["jugador"] + " (" + base["rol"].astype(str) + ")"
        combos = sorted(base["jug_rol"].unique().tolist())
        sel_combo = st.multiselect("Jugador (Rol)", combos, default=combos[:min(6, len(combos))])
        base = base[base["jug_rol"].isin(sel_combo)]
        base = base[base["minutos"] >= min_minutos].copy()
        label_col = "jug_rol"

    if base.empty:
        st.warning("No hay filas luego de aplicar filtros.")
        st.stop()

    # validar que estén las columnas
    miss = [m for m in metrics if m not in base.columns]
    if miss:
        st.error(f"Faltan columnas en los datos: {miss}")
        st.stop()

    # ---------- RADAR ----------
    # construir tabla radar (0–1 para % y reescalado para absolutos si querés)
    # regla: % se pasa a 0–1; absolutos se normalizan min-max sobre el subconjunto para comparabilidad de forma.
    rad = base[[label_col, "minutos"] + metrics].copy()

    # % → 0–1
    for c in metrics:
        if "%" in c:
            rad[c] = pd.to_numeric(rad[c], errors="coerce") / 100.0

    # abs → min-max (subconjunto actual), evitando división por 0
    for c in metrics:
        if "%" not in c:
            s = pd.to_numeric(rad[c], errors="coerce")
            mn, mx = float(np.nanmin(s.values)), float(np.nanmax(s.values))
            if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                rad[c] = (s - mn) / (mx - mn)
            else:
                # si todos iguales → 0.5 para que se vea, o 0
                rad[c] = 0.5

    labels = metrics
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.close("all")
    fig = plt.figure(figsize=(9.5, 9.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=9)

    series = rad[label_col].tolist()
    vals_mat = rad[labels].values

    # paleta tab10/tab20
    n_series = len(series)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_series,10))) if n_series <= 10 else plt.cm.tab20(np.linspace(0, 1, min(n_series,20)))

    for i, row in enumerate(vals_mat):
        vals = row.tolist()
        vals += vals[:1]
        color = colors[i % len(colors)]
        tag = f"{series[i]} ({int(round(rad.iloc[i]['minutos']))}m)"
        ax.plot(angles, vals, linewidth=2.0, marker="o", markersize=4, color=color, label=tag)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ttl = "Radar — Jugador total" if scope=="Jugador total" else ("Radar — Por rol" if scope=="Por rol" else "Radar — Jugador & Rol")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=9, frameon=False)
    plt.title(ttl, fontsize=14, pad=18)
    st.pyplot(fig, use_container_width=True)

    # ---------- Tabla de valores visibles ----------
    # Para la tabla: % en 0–100 (si quieres), absolutos mostrar sin min-max (entonces mostramos la tabla original base)
    tabla = base[[label_col, "minutos"] + metrics].copy()
    for c in metrics:
        if "%" in c:
            tabla[c] = pd.to_numeric(tabla[c], errors="coerce").round(2)
        else:
            tabla[c] = pd.to_numeric(tabla[c], errors="coerce").round(2)

    st.markdown("**Tabla de métricas (promedios; abs normalizados a 40’ por partido antes de promediar)**")
    st.dataframe(tabla.sort_values("minutos", ascending=False), use_container_width=True)

# =========================
# 📋 TABLA & RESULTADOS
# =========================
if menu == "📋 Tabla & Resultados":
    import requests, pandas as pd, numpy as np
    from datetime import datetime
    import time

    st.subheader("📋 Tabla & Resultados")

    # --- Parámetros de la competencia (los que vos pasaste que funcionan) ---
    URL_TABLA   = "https://api.weball.me/public/tournament/176/phase/150/group/613/clasification?instanceUUID=2d260df1-7986-49fd-95a2-fcb046e7a4fb"
    URL_MATCHES = "https://api.weball.me/public/tournament/176/phase/150/matches?instanceUUID=2d260df1-7986-49fd-95a2-fcb046e7a4fb"
    HEADERS     = {"Content-Type": "application/json"}
    CATEG_FILTRO = "2016 PROMOCIONALES"     # incluimos SOLO esta categoría
    EXCLUIR_LIBRE = True

    # ---------- Helpers ----------
    @st.cache_data(ttl=300, show_spinner=False)
    def _safe_get(url, max_tries=3, sleep_s=0.8):
        last_err = None
        for i in range(max_tries):
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(sleep_s)
        raise last_err

    @st.cache_data(ttl=300, show_spinner=False)
    def _safe_post(url, payload, max_tries=3, sleep_s=0.8):
        last_err = None
        for i in range(max_tries):
            try:
                r = requests.post(url, headers=HEADERS, json=payload, timeout=20)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(sleep_s)
        raise last_err

    @st.cache_data(ttl=300)
    def fetch_tabla():
        data = _safe_get(URL_TABLA)
        # La tabla correcta está en data[1]['positions'] (como mostraste)
        positions = data[1]['positions']
        equipos = []
        for equipo in positions:
            nombre = equipo['labelTablePosition']['name']
            pts = equipo['pts']; pj = equipo['pj']; pg = equipo['pg']; pe = equipo['pe']; pp = equipo['pp']
            gf = equipo['gf']; gc = equipo['gc']; dg = equipo['dg']
            equipos.append({"Equipo": nombre, "Pts": pts, "PJ": pj, "PG": pg, "PE": pe, "PP": pp, "GF": gf, "GC": gc, "DG": dg})
        df = pd.DataFrame(equipos)
        df = df.sort_values(["Pts","DG","GF"], ascending=[False,False,False]).reset_index(drop=True)
        df.insert(0, "Pos", df.index + 1)
        return df

    @st.cache_data(ttl=300)
    def fetch_partidos():
        # primera página para saber totalPages
        d0 = _safe_post(URL_MATCHES, {})
        total_pages = int(d0.get("totalPages", 1))
        partidos = []

        for page in range(1, total_pages + 1):
            d = _safe_post(URL_MATCHES, {"page": page})
            for match in d.get("data", []):
                categoria = (match.get("category", {})
                                   .get("categoryInstance", {})
                                   .get("name", "")) or ""
                if CATEG_FILTRO not in categoria:
                    continue

                # Equipos
                try:
                    local = match['clubHome']['clubInscription']['club']['name']
                    visitante = match['clubAway']['clubInscription']['club']['name']
                except Exception:
                    local, visitante = "LIBRE", ""

                if EXCLUIR_LIBRE and (local.upper() == "LIBRE" or visitante.upper() == "LIBRE"):
                    continue

                gl = match.get("scoreHome")
                gv = match.get("scoreAway")

                status = match.get("status")
                estado = status.get("label") if isinstance(status, dict) else "Desconocido"

                contenedor = match.get("containerItemsId", [])
                jornada_id = contenedor[0] if contenedor else None

                fecha_tecnica = match.get("updatedAt") or match.get("createdAt")

                partidos.append({
                    "Fecha Técnica": fecha_tecnica,
                    "Jornada ID": jornada_id,
                    "Estado": str(estado),
                    "Equipo Local": local,
                    "Equipo Visitante": visitante,
                    "Goles Local": gl,
                    "Goles Visitante": gv
                })

        df = pd.DataFrame(partidos)
        if df.empty:
            return df, pd.DataFrame(), pd.DataFrame()

        # mapear Jornada ID -> Fecha N SÓLO para finalizados
        ids_ordenados = sorted(df["Jornada ID"].dropna().unique())
        mapa_fechas = {jid: f"Fecha {i+1}" for i, jid in enumerate(ids_ordenados)}
        df["Fecha"] = None
        fin_mask = df["Estado"].str.lower().eq("finalizado")
        df.loc[fin_mask, "Fecha"] = df.loc[fin_mask, "Jornada ID"].map(mapa_fechas)

        # Partidos jugados / fixture
        df_res = df[fin_mask].copy()
        df_fix = df[~df["Estado"].str.lower().isin(["finalizado","cancelado"])].copy()

        # cast fechas
        for dff in (df, df_res, df_fix):
            dff["Fecha Técnica"] = pd.to_datetime(dff["Fecha Técnica"], errors="coerce")

        return df, df_res, df_fix

    # ---------- Rachas por equipo (en fechas jugadas) ----------
    def build_streaks(df_resultados: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve racha (G×n / E×n / P×n) SOLO en fechas jugadas por equipo (sin fill).
        """
        if df_resultados is None or df_resultados.empty:
            return pd.DataFrame(columns=["Fecha Técnica","Equipo","Racha"])

        df = df_resultados.copy()
        df["Fecha Técnica"] = pd.to_datetime(df["Fecha Técnica"], errors="coerce")
        df = df.sort_values("Fecha Técnica")

        rows = []
        for _, r in df.iterrows():
            gl = int(r["Goles Local"]); gv = int(r["Goles Visitante"])
            f  = r["Fecha Técnica"]
            if gl > gv:
                rows.append((f, r["Equipo Local"], "G"))
                rows.append((f, r["Equipo Visitante"], "P"))
            elif gl < gv:
                rows.append((f, r["Equipo Local"], "P"))
                rows.append((f, r["Equipo Visitante"], "G"))
            else:
                rows.append((f, r["Equipo Local"], "E"))
                rows.append((f, r["Equipo Visitante"], "E"))

        hist = pd.DataFrame(rows, columns=["Fecha Técnica","Equipo","Res"]).sort_values(["Equipo","Fecha Técnica"])

        out = []
        for eq, g in hist.groupby("Equipo"):
            last = None; streak = 0
            for _, rr in g.iterrows():
                res = rr["Res"]
                if res == last:
                    streak += 1
                else:
                    last = res
                    streak = 1
                out.append({"Fecha Técnica": rr["Fecha Técnica"], "Equipo": eq, "Racha": f"{res}×{streak}"})

        return pd.DataFrame(out).sort_values(["Equipo","Fecha Técnica"])

    # ---------- Evolución: tabla fecha a fecha + movimiento + racha (forward-fill) ----------
    def build_tabla_evolutiva(df_resultados: pd.DataFrame) -> pd.DataFrame:
        """
        Construye snapshots de tabla luego de cada partido finalizado.
        Agrega Posición, Movimiento vs snapshot anterior (▲/▼/＝) y Racha forward-filled.
        """
        if df_resultados is None or df_resultados.empty:
            return pd.DataFrame()

        # Orden cronológico
        df = df_resultados.copy().sort_values("Fecha Técnica").reset_index(drop=True)

        # Equipos
        equipos = pd.Series(df["Equipo Local"].tolist() + df["Equipo Visitante"].tolist()).unique()
        stats = {e: {"Pts":0,"PJ":0,"PG":0,"PE":0,"PP":0,"GF":0,"GC":0} for e in equipos}

        snapshots = []

        # armamos snapshot luego de CADA partido finalizado (orden real por fecha técnica)
        for _, row in df.iterrows():
            local = row["Equipo Local"]; visitante = row["Equipo Visitante"]
            gl = int(row["Goles Local"]); gv = int(row["Goles Visitante"])
            ftec = row["Fecha Técnica"]

            # PJ
            stats[local]["PJ"] += 1; stats[visitante]["PJ"] += 1
            # Goles
            stats[local]["GF"] += gl; stats[local]["GC"] += gv
            stats[visitante]["GF"] += gv; stats[visitante]["GC"] += gl
            # Puntos
            if gl > gv:
                stats[local]["Pts"] += 3; stats[local]["PG"] += 1; stats[visitante]["PP"] += 1
            elif gl < gv:
                stats[visitante]["Pts"] += 3; stats[visitante]["PG"] += 1; stats[local]["PP"] += 1
            else:
                stats[local]["Pts"] += 1; stats[visitante]["Pts"] += 1
                stats[local]["PE"] += 1; stats[visitante]["PE"] += 1

            # snapshot ordenado
            tabla = []
            for e in equipos:
                tabla.append({
                    "Fecha Técnica": ftec,
                    "Equipo": e,
                    "Pts": stats[e]["Pts"],
                    "PJ": stats[e]["PJ"],
                    "PG": stats[e]["PG"],
                    "PE": stats[e]["PE"],
                    "PP": stats[e]["PP"],
                    "GF": stats[e]["GF"],
                    "GC": stats[e]["GC"],
                    "DG": stats[e]["GF"] - stats[e]["GC"]
                })
            snap = pd.DataFrame(tabla).sort_values(["Pts","DG","GF"], ascending=[False,False,False]).reset_index(drop=True)
            snap["Posición"] = snap.index + 1
            snapshots.append(snap)

        evo = pd.concat(snapshots, ignore_index=True)

        # Movimiento vs snapshot anterior por equipo
        evo = evo.sort_values(["Fecha Técnica","Equipo"])
        evo["Movimiento"] = "＝ 0"
        for e, g in evo.groupby("Equipo"):
            g = g.sort_values("Fecha Técnica")
            mov = ["＝ 0"]
            for i in range(1, len(g)):
                delta = int(g.iloc[i-1]["Posición"]) - int(g.iloc[i]["Posición"])
                if delta > 0:
                    mov.append(f"▲ +{delta}")
                elif delta < 0:
                    mov.append(f"▼ {delta}")
                else:
                    mov.append("＝ 0")
            evo.loc[g.index, "Movimiento"] = mov

        # Racha (jugadas) + forward-fill a TODOS los snapshots
        df_racha_jug = build_streaks(df)
        calendar = (evo[["Fecha Técnica","Equipo"]]
                    .drop_duplicates()
                    .sort_values(["Equipo","Fecha Técnica"]))
        df_racha_full = calendar.merge(df_racha_jug, on=["Fecha Técnica","Equipo"], how="left")
        df_racha_full["Racha"] = (df_racha_full
                                  .sort_values(["Equipo","Fecha Técnica"])
                                  .groupby("Equipo")["Racha"]
                                  .ffill()
                                  .fillna("—"))  # antes del debut

        evo = evo.merge(df_racha_full, on=["Fecha Técnica","Equipo"], how="left")
        return evo.sort_values(["Fecha Técnica","Posición"])

    # ---------- UI ----------
    tabs = st.tabs(["🏆 Tabla actual", "📊 Resultados", "📅 Próximos", "📈 Tabla fecha a fecha"])

    # TABLA ACTUAL
    with tabs[0]:
        try:
            df_tabla = fetch_tabla()
            st.dataframe(
                df_tabla,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pos": st.column_config.NumberColumn("Pos", width="small"),
                    "Pts": st.column_config.NumberColumn("Pts", width="small"),
                    "PJ": st.column_config.NumberColumn("PJ", width="small"),
                    "PG": st.column_config.NumberColumn("PG", width="small"),
                    "PE": st.column_config.NumberColumn("PE", width="small"),
                    "PP": st.column_config.NumberColumn("PP", width="small"),
                    "GF": st.column_config.NumberColumn("GF", width="small"),
                    "GC": st.column_config.NumberColumn("GC", width="small"),
                    "DG": st.column_config.NumberColumn("DG", width="small"),
                }
            )
        except Exception as e:
            st.error(f"Error consultando tabla: {e}")

    # RESULTADOS & FIXTURE (y base para evolutiva)
    try:
        df_all, df_resultados, df_fixture = fetch_partidos()
    except Exception as e:
        df_resultados = pd.DataFrame(); df_fixture = pd.DataFrame()
        st.error(f"Error consultando partidos: {e}")

    # RESULTADOS
    with tabs[1]:
        if df_resultados.empty:
            st.info("No hay resultados disponibles.")
        else:
            df_view = df_resultados.sort_values("Fecha Técnica", ascending=True).copy()
            df_view["Fecha Técnica"] = df_view["Fecha Técnica"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                df_view[["Fecha","Fecha Técnica","Equipo Local","Goles Local","Goles Visitante","Equipo Visitante"]],
                use_container_width=True, hide_index=True
            )

    # PRÓXIMOS
    with tabs[2]:
        if df_fixture.empty:
            st.info("No hay próximos partidos publicados.")
        else:
            df_fix_view = df_fixture.sort_values("Fecha Técnica", ascending=True).copy()
            df_fix_view["Fecha Técnica"] = df_fix_view["Fecha Técnica"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                df_fix_view[["Fecha Técnica","Equipo Local","Equipo Visitante","Estado"]],
                use_container_width=True, hide_index=True
            )

    # TABLA FECHA A FECHA
    with tabs[3]:
        if df_resultados.empty:
            st.info("Necesito resultados finalizados para construir la evolución.")
        else:
            evo = build_tabla_evolutiva(df_resultados)

            # Selector de corte temporal (snapshot más cercano a la Fecha seleccionada)
            fechas_disp = evo["Fecha Técnica"].dropna().sort_values().unique()
            sel_fecha = st.select_slider("Corte temporal", options=list(fechas_disp), value=fechas_disp[-1],
                                         format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d %H:%M"))

            snap = (evo[evo["Fecha Técnica"] <= pd.to_datetime(sel_fecha)]
                        .sort_values(["Fecha Técnica"])
                        .groupby("Equipo", as_index=False).tail(1))

            snap = snap.sort_values(["Pts","DG","GF"], ascending=[False,False,False]).reset_index(drop=True)
            snap.insert(0, "Pos", snap.index + 1)

            # Orden amigable columnas
            cols = ["Pos","Equipo","Pts","PJ","PG","PE","PP","GF","GC","DG","Movimiento","Racha"]
            cols = [c for c in cols if c in snap.columns]
            st.dataframe(
                snap[cols],
                use_container_width=True, hide_index=True
            )

            st.caption("Movimiento: comparación contra el snapshot inmediato anterior (▲ sube / ▼ baja / ＝ igual).  Racha: última secuencia viva (G/E/P).")



