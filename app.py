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
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Arc, Circle as MplCircle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import seaborn as sns
from collections import Counter, defaultdict
from functools import lru_cache

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
# --- Tamaños/zoom de escudos (ajustables rápido) ---
LOGO_Z_ELO        = 0.08
LOGO_Z_WDL_SINGLE = 0.07
LOGO_Z_WDL_DOUBLE = 0.06
WDL_MAX_LOGOS     = 18   # si hay más puntos que esto, usamos puntos en vez de logos

# Tamaños de escudos (ajustables)
LOGO_PX_ELO_DEFAULT = 14      # antes 32 (ELO más chico)
# --- Ajustes de tamaño/espaciado de logos (¡todos iguales y chicos!) ---
LOGO_PX_ELO         = 12   # tamaño en píxeles del logo en el gráfico ELO
LOGO_PX_WDL_SINGLE  = 14   # tamaño en píxeles (W/D/L con 1 panel)
LOGO_PX_WDL_DOUBLE  = 12   # tamaño en píxeles (W/D/L con 2 paneles)
RIGHT_MARGIN_ELO    = 0.40 # margen a la derecha para encajar logos en ELO (0.30–0.55)
MIN_GAP_ELO         = 10.0  # separación mínima vertical entre logos en ELO


BANNER_H   = 0.145
LOGO_W     = 0.105
TITLE_FS   = 32
SUB_FS     = 19

FOOTER_H        = 0.120
FOOTER_LOGO_W   = 0.105
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

# === Normalización y alias de nombres de equipos ===
def _norm_key(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", " ", s)

# Si tu filename no coincide 1:1 con el nombre de la tabla, mapealo acá.
BADGE_ALIASES = {
    "FERRO CARRIL OESTE": "ferro",                # si el archivo es images/equipos/ferro.png
    "VELEZ SARSFIELD":   "velez sarsfield",
    "UNION EZPELETA":    "union ezpeleta",
    "SAN LORENZO":       "san lorenzo",
    "RIVER PLATE":       "river plate",
    "BOCA JUNIORS":      "boca juniors",
    "SOCIEDAD HEBRAICA": "sociedad hebraica",
    "FRANJA DE ORO":     "franja de oro",
    "INDEPENDIENTE (A)": "independiente a",
    "MIRIÑAQUE":         "mirinaque",             # por la ñ → n, si tu archivo está sin tilde
    "17 DE AGOSTO":      "17 de agosto",
    "KIMBERLEY":         "kimberley",
    "PINOCHO":           "pinocho",
    "VILLA MODELO":      "villa modelo",
    "CISSAB":            "cissab",
}

_BADGE_INDEX: dict[str, str] | None = None

def _build_badge_index() -> dict[str, str]:
    idx = {}
    if not os.path.isdir(BADGE_DIR):
        return idx
    exts = ("*.png","*.jpg","*.jpeg","*.webp","*.svg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(BADGE_DIR, e)))
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        k = _norm_key(stem)
        idx[k] = p
        # Variantes típicas
        idx[_norm_key(stem.replace("_"," "))] = p
        idx[_norm_key(stem.replace("-"," "))] = p
        idx[_norm_key(re.sub(r"\bclub\b","", stem))] = p
    # Aplicar ALIASES si el destino existe en el index
    for src, dst in BADGE_ALIASES.items():
        dst_key = _norm_key(dst)
        if dst_key in idx:
            idx[_norm_key(src)] = idx[dst_key]
    return idx

def badge_path_for(name: str) -> Optional[str]:
    """Devuelve el path del escudo para 'name' buscando en images/equipos."""
    global _BADGE_INDEX
    if _BADGE_INDEX is None:
        _BADGE_INDEX = _build_badge_index()
    if not name:
        return None
    k = _norm_key(name)
    if k in _BADGE_INDEX:
        return _BADGE_INDEX[k]
    # Fallback suave: substring match
    for kk, path in _BADGE_INDEX.items():
        if k in kk or kk in k:
            return path
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
    """
    Lee del XML los códigos de posesión escritos como:
      - 'Tiempo Posesión Ferro' / 'Tiempo Posecion Ferro'
      - 'Tiempo Posesión Rival' / 'Tiempo Posecion Rival'
    (con o sin tilde, mayúsculas, espacios extra).
    Devuelve (posesión_ferro_pct, posesión_rival_pct).
    """
    if not xml_path or not os.path.isfile(xml_path):
        return 0.0, 0.0

    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return 0.0, 0.0

    t_ferro = 0.0
    t_rival = 0.0

    for inst in root.findall(".//instance"):
        # normaliza: sin tildes, minúscula y colapsa espacios
        raw = inst.findtext("code") or ""
        code = re.sub(r"\s+", " ", nlower(raw)).strip()

        # duraciones
        try:
            stt = float(inst.findtext("start") or "0")
            enn = float(inst.findtext("end") or "0")
        except Exception:
            continue
        dur = max(0.0, enn - stt)

        # acepta 'posesion' y 'posecion'
        if code in {"tiempo posesion ferro", "tiempo posecion ferro"}:
            t_ferro += dur
        elif code in {"tiempo posesion rival", "tiempo posecion rival"}:
            t_rival += dur

    tot = t_ferro + t_rival
    if tot <= 0:
        return 0.0, 0.0

    return round(100.0 * t_ferro / tot, 1), round(100.0 * t_rival / tot, 1)

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

    # --- filas (quién) -------------------------------------------------------
    who = df[player_col].fillna("").astype(str).map(nlower)

    EXCL_PREFIXES = (
        "tiempo posecion ferro", "tiempo posesion ferro",
        "tiempo posecion rival", "tiempo posesion rival",
        "tiempo no jugado",
    )

    is_real = who.ne("")
    is_total_row = who.str.match(r"^totales?$")
    is_excluded = is_total_row | who.str.startswith(EXCL_PREFIXES)

    is_rival = is_real & who.eq("categoria - equipo rival")
    is_mine  = is_real & ~is_rival & ~is_excluded

    # --- helper: sumar columnas por nombres normalizados ---------------------
    # header_map: {col_original -> nombre_normalizado}
    from collections import defaultdict
    norm2cols = defaultdict(list)
    for c in df.columns[1:]:
        norm2cols[header_map[c]].append(c)

    def _expand_variants(name: str) -> set[str]:
        """tolerar 'progesivo' además de 'progresivo'"""
        n = nlower(name)
        out = {n}
        if "progresivo" in n:
            out.add(n.replace("progresivo", "progesivo"))
        if "progresiva" in n:
            out.add(n.replace("progresiva", "progesiva"))
        return out

    def sum_by_names(mask, names: list[str]) -> int:
        # junta columnas por nombres normalizados
        targets = set()
        for nm in names:
            targets |= _expand_variants(nlower(nm))
        cols = []
        for t in targets:
            cols += norm2cols.get(t, [])
    
        if not cols:
            return 0
    
        block = df.loc[mask, cols]
    
        # ya tenemos números por cómo carga load_matrix(), pero
        # hacemos la suma de forma segura para Serie o DataFrame
        if isinstance(block, pd.Series):
            return int(pd.to_numeric(block, errors="coerce").fillna(0).sum())
        else:
            block_num = block.apply(pd.to_numeric, errors="coerce").fillna(0)
            return int(block_num.sum(numeric_only=True).sum())


    # --- listas exactas (según tu definición) --------------------------------
    PASSES_BASE = [
        "Pase Progresivo Frontal",
        "Pase Progresivo Lateral",
        "Pase Corto Frontal",
        "Pase Corto Lateral",
        "Pase Progresivo Frontal cPie",
        "Pase Progresivo Lateral cPie",
        "Salida de arco progresivo cMano",
        "Pase Corto Frontal cPie",
        "Pase Corto Lateral cPie",
        "Salida de arco corto cMano",
    ]
    PASSES_OK = [
        "Pase Progresivo Frontal Completado",
        "Pase Progresivo Lateral Completado",
        "Pase Corto Frontal Completado",
        "Pase Corto Lateral Completado",
        "Pase Progresivo Frontal Completado cPie",
        "Pase Progresivo Lateral Completado cPie",
        "Salida de arco progresivo Completado cMano",
        "Pase Corto Frontal Completado cPie",
        "Pase Corto Lateral Completado cPie",
        "Salida de arco corto Completado cMano",
    ]

    RECUP_LIST = [
        "Recuperacion x Duelo",
        "Recuperación x Interceptacion",
        "Recuperacion x Robo",
        "Recuperacion x Mal Pase Rival",
        "Recuperacion x Mal Control",
    ]
    CORNERS_LIST = ["Corner En corto", "Corner Al Area", "Corner 2do Palo"]

    # algunas columnas pueden venir en singular/plural
    TIRO_HECHO   = ["Tiro Hecho"]
    TIRO_ARCO    = ["Tiro al arco", "Tiros al arco"]  # tolerancia

    UNO_VS_UNO_G = ["1v1 Ganado"]

    # --- FERRO ---------------------------------------------------------------
    pases_base_m = sum_by_names(is_mine, PASSES_BASE)
    pases_ok_m   = sum_by_names(is_mine, PASSES_OK)
    tiros_m      = sum_by_names(is_mine, TIRO_HECHO)
    tiros_arco_m = sum_by_names(is_mine, TIRO_ARCO)

    recup_m      = sum_by_names(is_mine, RECUP_LIST)
    rec_duelo_m  = sum_by_names(is_mine, ["Recuperacion x Duelo"])
    uno_v_uno_m  = sum_by_names(is_mine, UNO_VS_UNO_G)
    duelos_g_m   = rec_duelo_m + uno_v_uno_m

    corners_m    = sum_by_names(is_mine, CORNERS_LIST)
    faltas_m     = sum_by_names(is_mine, ["Faltas Recibidas"])
    goles_m      = sum_by_names(is_mine, ["Gol"])
    asis_m       = sum_by_names(is_mine, ["Asistencia"])
    pclave_m     = sum_by_names(is_mine, ["Pase Clave"])

    # --- RIVAL ---------------------------------------------------------------
    pases_base_r = sum_by_names(is_rival, PASSES_BASE)
    pases_ok_r   = sum_by_names(is_rival, PASSES_OK)
    tiros_r      = sum_by_names(is_rival, TIRO_HECHO)
    tiros_arco_r = sum_by_names(is_rival, TIRO_ARCO)

    recup_r      = sum_by_names(is_rival, RECUP_LIST)
    rec_duelo_r  = sum_by_names(is_rival, ["Recuperacion x Duelo"])
    uno_v_uno_r  = sum_by_names(is_rival, UNO_VS_UNO_G)
    duelos_g_r   = rec_duelo_r + uno_v_uno_r

    corners_r    = sum_by_names(is_rival, CORNERS_LIST)
    faltas_r     = sum_by_names(is_rival, ["Faltas Recibidas"])
    goles_r      = sum_by_names(is_rival, ["Gol"])
    asis_r       = sum_by_names(is_rival, ["Asistencia"])
    pclave_r     = sum_by_names(is_rival, ["Pase Clave"])

    # --- %s ------------------------------------------------------------------
    def pct(a, b):
        return round(100.0 * a / b, 1) if b else 0.0

    pases_ok_pct_m = pct(pases_ok_m, pases_base_m)
    pases_ok_pct_r = pct(pases_ok_r, pases_base_r)

    duelos_pct_m   = pct(duelos_g_m, duelos_g_m + duelos_g_r)
    duelos_pct_r   = pct(duelos_g_r, duelos_g_m + duelos_g_r)

    # --- salida con las CLAVES DEL PANEL (ROW_ORDER) ------------------------
    return {
        "FERRO": {
            "Pases totales": pases_base_m,
            "Pases OK %":    pases_ok_pct_m,
            "Tiros":         tiros_m,
            "Tiros al arco": tiros_arco_m,
            "Recuperaciones": recup_m,
            "Duelos ganados": duelos_g_m,
            "% Duelos ganados": duelos_pct_m,
            "Corners":       corners_m,
            "Faltas":        faltas_m,
            "Goles":         goles_m,
            "Asistencias":   asis_m,
            "Pases clave":   pclave_m,
        },
        "RIVAL": {
            "Pases totales": pases_base_r,
            "Pases OK %":    pases_ok_pct_r,
            "Tiros":         tiros_r,
            "Tiros al arco": tiros_arco_r,
            "Recuperaciones": recup_r,
            "Duelos ganados": duelos_g_r,
            "% Duelos ganados": duelos_pct_r,
            "Corners":       corners_r,
            "Faltas":        faltas_r,
            "Goles":         goles_r,
            "Asistencias":   asis_r,
            "Pases clave":   pclave_r,
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

    # ---------- Banner superior (idéntico a Estadísticas) ----------
    BANNER_Y0 = 1.0 - (BANNER_H + 0.02)
    ax.add_patch(Rectangle((0, BANNER_Y0), 1, BANNER_H, facecolor="white", edgecolor="none", zorder=5))

    if ferro_logo_path: draw_logo(ax, ferro_logo_path, 0.09, BANNER_Y0 + BANNER_H*0.52, LOGO_W)
    if rival_logo_path: draw_logo(ax, rival_logo_path, 0.91, BANNER_Y0 + BANNER_H*0.52, LOGO_W)

    ax.text(0.5, BANNER_Y0 + BANNER_H*0.63, f"FERRO vs {rival_name.upper()}",
            ha="center", va="center", fontsize=TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, BANNER_Y0 + BANNER_H*0.29, "TIMELINE",
            ha="center", va="center", fontsize=SUB_FS,   weight="bold", color=bg_green, zorder=7)

    # ---------- Banner inferior (idéntico a Estadísticas) ----------
    FOOTER_Y0 = 0.02
    ax.add_patch(Rectangle((0, FOOTER_Y0), 1, FOOTER_H, facecolor="white", edgecolor="none", zorder=5))

    if os.path.isfile(FOOTER_LEFT_LOGO):
        draw_logo(ax, FOOTER_LEFT_LOGO,  0.09, FOOTER_Y0 + FOOTER_H*0.52, FOOTER_LOGO_W)
    if os.path.isfile(FOOTER_RIGHT_LOGO):
        draw_logo(ax, FOOTER_RIGHT_LOGO, 0.91, FOOTER_Y0 + FOOTER_H*0.52, FOOTER_LOGO_W)

    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.63, "Trabajo Fin de Máster",
            ha="center", va="center", fontsize=FOOTER_TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.28, "Cristian Dieguez",
            ha="center", va="center", fontsize=FOOTER_SUB_FS,   weight="bold", color=bg_green, zorder=7)

    # ---------- Cuerpo del timeline (misma lógica, solo ajusta márgenes) ----------
    EXTRA_GAP_BELOW_BANNER = 0.075
    panel_y0 = FOOTER_Y0 + FOOTER_H + 0.012
    panel_y1 = BANNER_Y0 - EXTRA_GAP_BELOW_BANNER
    panel_h  = panel_y1 - panel_y0

    x_center_gap_L, x_center_gap_R = 0.47, 0.53
    x_bar_M_max, x_bar_R_max       = 0.22, 0.78
    x_shot_M, x_last_M             = 0.05, 0.16
    x_shot_R, x_last_R             = 0.95, 0.84
    x_goal_M, x_goal_R             = x_shot_M - 0.025, x_shot_R + 0.025

    # Títulos de columnas
    ty = panel_y1 + 0.012
    ax.text(x_last_M,  ty, "Últ. tercio", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_bar_M_max+x_center_gap_L)/2, ty, "Pases/min", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_shot_M,  ty, "Tiros / Goles", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_center_gap_L+x_center_gap_R)/2, ty, "Min.", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text((x_center_gap_R+x_bar_R_max)/2, ty, "Pases/min", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_shot_R,  ty, "Tiros / Goles", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(x_last_R,  ty, "Últ. tercio",   ha="center", va="bottom", fontsize=10, weight="bold")

    # Minutos y separadores
    M = 40
    def y_of_index(i): return panel_y1 - panel_h * (i + 0.5) / M
    ax.add_line(plt.Line2D([x_center_gap_L, x_center_gap_L], [panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    ax.add_line(plt.Line2D([x_center_gap_R, x_center_gap_R], [panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    for m in range(0, 41, 5):
        yy = y_of_index(min(m, 39))
        ax.text(0.50, yy, f"{m:02d}'", ha="center", va="center", fontsize=8, alpha=0.90)

    max_bar = max(int(tl["passes_M"].max() if tl["passes_M"].size else 1),
                  int(tl["passes_R"].max() if tl["passes_R"].size else 1), 1)

    def bar_width_left(cnt):  return (x_center_gap_L - x_bar_M_max) * (cnt / max_bar if max_bar else 0)
    def bar_width_right(cnt): return (x_bar_R_max - x_center_gap_R) * (cnt / max_bar if max_bar else 0)

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

# --- Descriptores permitidos ---
DESC_CANON = [
    "Valla Invicta en cancha",
    "Goles a favor en cancha",
    "Participa en Gol Hecho",
    "Gol Rival en cancha",
    "Involucrado en gol recibido",
]
DESC_CANON_LC = [nlower(x) for x in DESC_CANON]
DESC_IGNORE   = {nlower("Total")}  # label a ignorar si aparece solo

def cargar_minutos_desde_xml_totalvalues(xml_path: str) -> pd.DataFrame:
    """
    Lee SOLO instancias de Jugador (Rol) del XML TotalValues cuya lista de labels:
      - esté vacía (o sólo tenga 'Total'),  O
      - contenga al menos uno de estos descriptores:
        'Valla invicta en cancha', 'Goles a favor en cancha',
        'Participa en Gol Hecho', 'Gol Rival en cancha', 'Involucrado en gol recibido'

    Devuelve filas crudas de tramos: code, nombre, rol, start_s, end_s, dur_s
    + una columna por descriptor con 0/1 indicando si ese tramo lo contiene.
    """
    base_cols = ["code","nombre","rol","start_s","end_s","dur_s"]
    desc_cols = DESC_CANON[:]  # usamos los nombres canónicos como encabezados
    cols = base_cols + desc_cols

    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=cols)

    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return pd.DataFrame(columns=cols)

    rows = []
    for inst in root.findall(".//instance"):
        code = inst.findtext("code") or ""
        if not is_player_code(code):
            continue

        # labels normalizados
        labels = [(lab.findtext("text") or "").strip() for lab in inst.findall("./label")]
        labels_low = [nlower(l) for l in labels if l]
        decision = [l for l in labels_low if l not in DESC_IGNORE]

        # criterio de inclusión
        keep = (len(decision) == 0) or any(l in DESC_CANON_LC for l in decision)
        if not keep:
            continue

        # tiempos
        stt, enn = inst.findtext("start"), inst.findtext("end")
        try:
            s = float(stt) if stt is not None else None
            e = float(enn) if enn is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s:
            continue

        m = _NAME_ROLE_RE.match(code)
        if not m:
            continue
        nombre, rol = m.group(1).strip(), m.group(2).strip()

        # flags de descriptores
        present_lc = set(decision)
        flags = []
        for canon, canon_lc in zip(DESC_CANON, DESC_CANON_LC):
            flags.append(1 if canon_lc in present_lc else 0)

        rows.append({
            "code": code, "nombre": nombre, "rol": rol,
            "start_s": s, "end_s": e, "dur_s": e - s,
            **{canon: flag for canon, flag in zip(DESC_CANON, flags)}
        })

    df = pd.DataFrame(rows, columns=cols)
    return df

def _sumar_descriptores(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=by + DESC_CANON)
    # Asegura que las cols existan (por si algún XML no trae ninguno)
    for c in DESC_CANON:
        if c not in df.columns:
            df[c] = 0
    agg = df.groupby(by, as_index=False)[DESC_CANON].sum()
    return agg

def _prep_minutes_table(df: pd.DataFrame, include_role: bool=False) -> pd.DataFrame:
    """
    Deja la tabla con:
      - 'minutos' (texto MM:SS) a partir de 'mmss'
      - 'nT' (antes n_tramos)
      - sin 'segundos' ni 'minutos' numérico
      - columnas de descriptores al final (si no existen, las crea en 0)
    """
    if df is None or df.empty:
        base_cols = (["nombre","rol"] if include_role else ["nombre"]) + ["minutos","nT"] + DESC_CANON
        return pd.DataFrame(columns=base_cols)

    d = df.copy()

    # Asegurar cols descriptor
    for c in DESC_CANON:
        if c not in d.columns:
            d[c] = 0

    # Drop numéricas que no querés ver
    drop_cols = [c for c in ["segundos","minutos"] if c in d.columns]
    if drop_cols:
        d = d.drop(columns=drop_cols)

    # Renombrar y ordenar
    if "mmss" in d.columns:
        d = d.rename(columns={"mmss": "minutos"})
    if "n_tramos" in d.columns:
        d = d.rename(columns={"n_tramos": "nT"})

    keep = (["nombre","rol"] if include_role else ["nombre"]) + ["minutos","nT"] + DESC_CANON
    keep = [c for c in keep if c in d.columns]  # por si acaso
    d = d.loc[:, keep]

    # Evitar cualquier duplicado de nombres (defensivo)
    d = d.loc[:, ~d.columns.duplicated(keep="first")]
    return d

def _format_mmss(seconds: float | int) -> str:
    if seconds is None or not np.isfinite(seconds): return "00:00"
    s = int(round(float(seconds))); mm, ss = divmod(s, 60)
    return f"{mm:02d}:{ss:02d}"

def minutos_por_presencia(df_pres: pd.DataFrame):
    cols_rol = ["nombre","rol","segundos","mmss","minutos","n_tramos"]
    cols_jug = ["nombre","segundos","minutos","n_tramos","mmss"]

    if df_pres is None or df_pres.empty:
        return (pd.DataFrame(columns=cols_rol), pd.DataFrame(columns=cols_jug))

    out = []
    for (nombre, rol), g in df_pres.groupby(["nombre","rol"], dropna=False):
        intervals = list(zip(g["start_s"], g["end_s"]))
        merged = _merge_intervals(intervals)
        secs = sum(e - s for s, e in merged)
        out.append({
            "nombre": nombre, "rol": rol,
            "segundos": int(round(secs)),
            "mmss": _format_mmss(secs),
            "minutos": secs / 60.0,
            "n_tramos": len(merged)
        })

    df_por_rol = pd.DataFrame(out, columns=cols_rol)
    if not df_por_rol.empty:
        df_por_rol["minutos"] = df_por_rol["minutos"].round(2)
        df_por_rol = df_por_rol.sort_values(["segundos","nombre"], ascending=[False, True]).reset_index(drop=True)

        df_por_jugador = (df_por_rol.groupby("nombre", as_index=False)
                          .agg(segundos=("segundos","sum"),
                               minutos=("minutos","sum"),
                               n_tramos=("n_tramos","sum"))
                          .assign(mmss=lambda d: d["segundos"].apply(_format_mmss))
                          .sort_values(["segundos","nombre"], ascending=[False, True])
                          .reset_index(drop=True))
        df_por_jugador["minutos"] = df_por_jugador["minutos"].round(2)
    else:
        df_por_jugador = pd.DataFrame(columns=cols_jug)

    return df_por_rol, df_por_jugador

def _inside_bar_label(ax, bars, values_sec):
    xmax = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.0
    for b, secs in zip(bars, values_sec):
        mmss = _format_mmss(secs)
        w = b.get_width()  # minutos
        y = b.get_y() + b.get_height()/2

        if w < 0.60:
            x = min(w + 0.12, xmax * 0.98)
            ax.text(x, y, mmss, va="center", ha="left", fontsize=8, color="black", fontweight="bold")
        elif w < 1.40:
            ax.text(w * 0.98, y, mmss, va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.text(w * 0.98, y, mmss, va="center", ha="right", fontsize=9.5, color="white", fontweight="bold")

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

    # ---------- Banner superior ----------
    BANNER_Y0 = 1.0 - (BANNER_H + 0.02)
    ax.add_patch(Rectangle((0, BANNER_Y0), 1, BANNER_H, facecolor="white", edgecolor="none", zorder=5))

    # logos más chicos (usa LOGO_W)
    if ferro_logo_path: draw_logo(ax, ferro_logo_path, 0.09, BANNER_Y0 + BANNER_H*0.52, LOGO_W)
    if rival_logo_path: draw_logo(ax, rival_logo_path, 0.91, BANNER_Y0 + BANNER_H*0.52, LOGO_W)

    ax.text(0.5, BANNER_Y0 + BANNER_H*0.63, f"{home_name.upper()} vs {away_name.upper()}",
            ha="center", va="center", fontsize=TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, BANNER_Y0 + BANNER_H*0.29, "KEY STATS",
            ha="center", va="center", fontsize=SUB_FS, weight="bold", color=bg_green, zorder=7)

    # ---------- Banner inferior ----------
    FOOTER_Y0 = 0.02
    ax.add_patch(Rectangle((0, FOOTER_Y0), 1, FOOTER_H, facecolor="white", edgecolor="none", zorder=5))

    # ⬅️ Sport Data Campus (izq) — ➡️ Sevilla (der), igual que en Timeline
    if os.path.isfile(FOOTER_LEFT_LOGO):
        draw_logo(ax, FOOTER_LEFT_LOGO,  0.09, FOOTER_Y0 + FOOTER_H*0.52, FOOTER_LOGO_W)
    if os.path.isfile(FOOTER_RIGHT_LOGO):
        draw_logo(ax, FOOTER_RIGHT_LOGO, 0.91, FOOTER_Y0 + FOOTER_H*0.52, FOOTER_LOGO_W)

    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.63, "Trabajo Fin de Máster",
            ha="center", va="center", fontsize=FOOTER_TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.28, "Cristian Dieguez",
            ha="center", va="center", fontsize=FOOTER_SUB_FS,   weight="bold", color=bg_green, zorder=7)

    # ---------- Cuerpo ----------
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
# Helpers (Tablas & Gráficos)
# =========================

# --------- UI: mostrar tablas completas (sin scroll vertical) ----------
# --------- UI: mostrar tablas completas (sin scroll vertical) ----------
def show_full_table(df: pd.DataFrame, max_px: int = 1000):
    """
    Muestra el DataFrame completo sin índice y sin la columna 'Movimiento'.
    """
    import streamlit as st
    if df is None or df.empty:
        st.dataframe(pd.DataFrame(), use_container_width=True, height=200)
        return

    d = df.copy()
    if "Movimiento" in d.columns:
        d = d.drop(columns=["Movimiento"])

    # nos aseguramos de resetear índice para que no aparezca la col. 0..n
    d = d.reset_index(drop=True)

    n = int(len(d))
    row_h = 35   # alto aprox por fila
    hdr_h = 38
    height = max(140, min(max_px, hdr_h + n * row_h))

    # Streamlit >=1.31
    try:
        st.dataframe(d, use_container_width=True, height=height, hide_index=True)
    except TypeError:
        # fallback para versiones sin 'hide_index'
        st.dataframe(d.style.hide(axis="index"), use_container_width=True, height=height)


# --------- Jornada: índice 1..N a partir de "Jornada ID" ----------
def _build_jornada_index(df_res: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Añade columna JornadaN (1..N) a df_res usando el orden natural de Jornada ID.
    Devuelve (df_con_jornadaN, mapa_jid->N)
    """
    jid_order = sorted(pd.to_numeric(df_res["Jornada ID"], errors="coerce").dropna().unique())
    jid2n = {int(jid): i + 1 for i, jid in enumerate(jid_order)}

    d = df_res.copy()
    jid_num = pd.to_numeric(d["Jornada ID"], errors="coerce")
    d["JornadaN"] = jid_num.map(lambda x: jid2n.get(int(x)) if pd.notna(x) else np.nan).astype("Int64")
    return d, jid2n


# --------- ELO por jornada (snapshot tras cada fecha) ----------
def _compute_elo_by_jornada(df_res_cut: pd.DataFrame, base_elo: float = 1000.0, K: float = 24.0) -> pd.DataFrame:
    """
    Calcula ELO por jornada (1..max) y devuelve un DataFrame pivot:
      index = JornadaN ; columns = Equipo ; values = ELO.
    Si no hay datos, devuelve DataFrame vacío.
    """
    d = df_res_cut.copy()
    d = d.dropna(subset=["JornadaN", "Goles Local", "Goles Visitante"])
    if d.empty or not np.isfinite(pd.to_numeric(d["JornadaN"], errors="coerce")).any():
        return pd.DataFrame(index=pd.Index([], name="JornadaN"))

    teams = sorted(pd.unique(pd.concat([d["Equipo Local"], d["Equipo Visitante"]], ignore_index=True)))
    elo = {t: float(base_elo) for t in teams}
    rows = []

    max_j = int(np.nanmax(pd.to_numeric(d["JornadaN"], errors="coerce")))
    for j in range(1, max_j + 1):
        jj = d[d["JornadaN"] == j].sort_values("Fecha Técnica")
        if jj.empty:
            rows.append({"JornadaN": j, **elo})
            continue

        for _, r in jj.iterrows():
            h, a = r["Equipo Local"], r["Equipo Visitante"]
            gl, gv = int(r["Goles Local"]), int(r["Goles Visitante"])

            Ra, Rb = elo[h], elo[a]
            Ea = 1.0 / (1.0 + 10.0 ** ((Rb - Ra) / 400.0))
            if gl > gv:
                Sa, Sb = 1.0, 0.0
            elif gl < gv:
                Sa, Sb = 0.0, 1.0
            else:
                Sa, Sb = 0.5, 0.5
            elo[h] = Ra + K * (Sa - Ea)
            elo[a] = Rb + K * ((1.0 - Sa) - (1.0 - Ea))

        rows.append({"JornadaN": j, **elo})

    out = pd.DataFrame(rows).set_index("JornadaN")
    # suavizado ligero para evitar quiebres muy “rectos”
    out = out.rolling(window=2, min_periods=1).mean()
    return out


# --------- Plot: ELO por jornada (con etiquetas al final, sin leyenda lateral) ----------
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib.patheffects as pe

def plot_elo_por_jornada(elo_pivot: pd.DataFrame, equipos: list[str], max_j: int) -> plt.Figure:
    BG, GRID = "#E8F5E9", "#9E9E9E"
    right_margin = globals().get("RIGHT_MARGIN_ELO", 0.40)
    min_gap      = globals().get("MIN_GAP_ELO", 2.0)
    logo_px      = globals().get("LOGO_PX_ELO", 12)

    if elo_pivot is None or elo_pivot.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return fig

    if not equipos:
        equipos = list(elo_pivot.columns)

    # Trayectorias por equipo
    traj = {}
    for eq in equipos:
        if eq in elo_pivot.columns:
            s = elo_pivot[eq].dropna()
            s = s[s.index <= max_j]
            if not s.empty:
                traj[eq] = list(zip(s.index.tolist(), s.values.tolist()))

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(traj))))
    for i, (eq, pts) in enumerate(traj.items()):
        xs = [x for x,_ in pts]; ys = [y for _,y in pts]
        if xs:
            ax.plot(xs, ys, lw=2.0, color=palette[i % len(palette)], zorder=2)

    # Puntos finales (para ubicar logo)
    ends = []
    for i, (eq, pts) in enumerate(traj.items()):
        xs = [x for x,_ in pts]; ys = [y for _,y in pts]
        if xs:
            ends.append([eq, xs[-1], ys[-1], palette[i % len(palette)]])

    # Separación vertical mínima entre logos
    ends.sort(key=lambda t: t[2])
    for k in range(1, len(ends)):
        if ends[k][2] - ends[k-1][2] < min_gap:
            ends[k][2] = ends[k-1][2] + min_gap

    # Margen y ubicación de logos
    ax.set_xlim(0.5, max_j + right_margin)
    x_logo = max_j + right_margin - 0.06  # un pelito adentro del xlim

    # Logos (todos a LOGO_PX_ELO)
    for eq, _, y_end, col in ends:
        oi = _logo_image_for(eq, target_px=logo_px)
        if oi is not None:
            ab = AnnotationBbox(oi, (x_logo, y_end),
                                frameon=False, box_alignment=(0,0.5), pad=0.0,
                                zorder=4, clip_on=False)
            ax.add_artist(ab)
        else:
            ax.text(x_logo, y_end, eq.upper(),
                    va="center", ha="left", fontsize=8.6, color="#1F1F1F",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    zorder=3)

    ax.set_xticks(list(range(1, max_j + 1)))
    ax.set_xlabel("Fecha (Jornada)", fontsize=12, color="#1F1F1F")
    ax.set_ylabel("Índice ELO",     fontsize=12, color="#1F1F1F")
    ax.tick_params(colors="#1F1F1F")
    ax.grid(True, ls="--", lw=0.8, color=GRID, alpha=0.55)
    return fig

# --------- W/D/L por jornada (usando JornadaN=1..N) + Rival ----------
def build_wdl_por_jornada(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve (Equipo, Jornada, R, Rival) con R ∈ {'W','D','L'} y Rival el contrincante de esa jornada.
    Usa JornadaN (1..N). Si no existe, la crea con _build_jornada_index.
    """
    if df_res is None or df_res.empty:
        return pd.DataFrame(columns=["Equipo", "Jornada", "R", "Rival"])

    d = df_res.copy()
    if "JornadaN" not in d.columns:
        d, _ = _build_jornada_index(d)

    d = d.dropna(subset=["JornadaN", "Goles Local", "Goles Visitante"])
    if d.empty:
        return pd.DataFrame(columns=["Equipo", "Jornada", "R", "Rival"])

    rows = []
    for _, r in d.iterrows():
        j = int(r["JornadaN"])
        gl, gv = int(r["Goles Local"]), int(r["Goles Visitante"])
        loc, vis = r["Equipo Local"], r["Equipo Visitante"]

        if gl > gv:
            res_loc, res_vis = "W", "L"
        elif gl < gv:
            res_loc, res_vis = "L", "W"
        else:
            res_loc = res_vis = "D"

        rows.append({"Equipo": loc, "Jornada": j, "R": res_loc, "Rival": vis})
        rows.append({"Equipo": vis, "Jornada": j, "R": res_vis, "Rival": loc})

    out = pd.DataFrame(rows)
    return out.sort_values(["Equipo", "Jornada"]).reset_index(drop=True)


# --- alias LEGACY: cualquier llamada vieja seguirá funcionando ---
build_wdl_jornada = build_wdl_por_jornada

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

@lru_cache(maxsize=256)
def _badge_arr_cached(path: str, trim: bool=True, bg_tol: int=16):
    arr = load_any_image(path)
    if trim:
        arr = trim_margins(arr, bg_tol=bg_tol)
    return arr

def _logo_image_for(team: str, target_px: int = 12):
    p = badge_path_for(team)
    if not p:
        return None
    try:
        arr = load_any_image(p)
        if TRIM_LOGO_BORDERS:
            arr = trim_margins(arr, bg_tol=16)   # recorta bordes blancos si hay
        h = max(1, int(arr.shape[0]))           # alto real del png
        zoom = float(target_px) / float(h)       # hace TODOS = target_px
        return OffsetImage(arr, zoom=zoom, interpolation="lanczos")
    except Exception:
        return None

from matplotlib.offsetbox import AnnotationBbox

def plot_wdl_por_jornada(
    wdl_jornada_df: pd.DataFrame,
    eq_a: str,
    eq_b: str | None,
    max_j: int
) -> plt.Figure:
    BG = "#E8F5E9"; GRID = "#9E9E9E"
    COL_WIN, COL_DRAW, COL_LOSS = "#2E7D32", "#FBC02D", "#C62828"
    MAP_Y = {"L": 0, "D": 1, "W": 2}

    # tamaños de logo (con fallback si no definiste las globals)
    logo_px_single = int(globals().get("LOGO_PX_WDL_SINGLE", 12))
    logo_px_double = int(globals().get("LOGO_PX_WDL_DOUBLE", 10))

    def _prep(eq: str) -> pd.DataFrame:
        if not eq:
            return pd.DataFrame(columns=["Jornada","R","Rival","y"])
        d = (wdl_jornada_df[wdl_jornada_df["Equipo"] == eq]
             .sort_values("Jornada")
             .loc[:, ["Jornada","R","Rival"]]
             .dropna(subset=["Jornada","R"]))
        d = d[d["Jornada"] <= max_j]
        if d.empty:
            d = d.assign(y=[])
        else:
            d["y"] = d["R"].map(MAP_Y)
        return d

    d1 = _prep(eq_a)
    d2 = _prep(eq_b) if (eq_b and eq_b != "(ninguno)") else None

    fig, axes = plt.subplots(
        nrows=2 if d2 is not None else 1,
        ncols=1,
        figsize=(12, 6 if d2 is not None else 3.8),
        sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax_idx, (eq, dd) in enumerate([(eq_a, d1), (eq_b, d2)]):
        if dd is None:
            continue
        ax = axes[ax_idx]
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

        # =========================
        # FIX AL INDENTATION ERROR:
        # la línea de plot DEBE estar identada dentro del if
        # =========================
        if not dd.empty:
            ax.plot(dd["Jornada"], dd["y"], color="#455A64", lw=1.4, alpha=0.9, zorder=2)

            # logos chicos y uniformes
            target_px = logo_px_double if (d2 is not None) else logo_px_single
            for _, rr in dd.iterrows():
                oi = _logo_image_for(rr["Rival"], target_px=target_px)
                if oi is not None:
                    ab = AnnotationBbox(
                        oi, (rr["Jornada"], rr["y"]),
                        frameon=False, pad=0.0, zorder=4, clip_on=True
                    )
                    ax.add_artist(ab)
                else:
                    # fallback: punto coloreado
                    col = {"W": COL_WIN, "D": COL_DRAW, "L": COL_LOSS}[rr["R"]]
                    ax.scatter(
                        [rr["Jornada"]], [rr["y"]],
                        s=90, c=col, edgecolor="black", linewidths=0.7, zorder=3
                    )

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Derrota", "Empate", "Victoria"])
        ax.grid(True, axis="x", ls="--", lw=0.8, color=GRID, alpha=0.55)
        ax.set_xlim(0.5, max_j + 0.5)
        ax.set_ylim(-0.45, 2.45)
        ax.set_title(str(eq).upper(), pad=6, fontsize=12, color="#1F1F1F")
        ax.tick_params(colors="#1F1F1F")
        for spine in ax.spines.values():
            spine.set_color("#1F1F1F")

    axes[-1].set_xticks(list(range(1, max_j + 1)))
    axes[-1].set_xlabel("Fecha (Jornada)", fontsize=12, color="#1F1F1F")
    plt.tight_layout()
    return fig

def tabla_a_jornada(df_res: pd.DataFrame, j_corte: int) -> pd.DataFrame:
    """
    Construye la tabla tomando EXACTAMENTE los primeros j_corte partidos de cada equipo.
    (PJ parejos para todos)
    """
    if df_res is None or df_res.empty:
        return pd.DataFrame(columns=["Pos","Equipo","Pts","PJ","PG","PE","PP","GF","GC","DG","Racha"])

    # Aseguramos JornadaN
    d_j, _ = _build_jornada_index(df_res)
    d = d_j.dropna(subset=["JornadaN","Goles Local","Goles Visitante"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["Pos","Equipo","Pts","PJ","PG","PE","PP","GF","GC","DG","Racha"])

    # Lista de equipos
    equipos = pd.unique(pd.concat([d["Equipo Local"], d["Equipo Visitante"]], ignore_index=True))

    # helper resultado
    def _res(gl, gv):
        return "G" if gl>gv else ("E" if gl==gv else "P")

    rows = []
    rachas_dict = {eq: [] for eq in equipos}

    for eq in equipos:
        # Partidos del equipo ordenados por jornada
        m1 = d[d["Equipo Local"].eq(eq)]
        m2 = d[d["Equipo Visitante"].eq(eq)]
        dd = pd.concat([m1, m2], ignore_index=True).sort_values("JornadaN")

        # Tomamos exactamente los primeros j_corte (si tiene menos, serán menos)
        dd = dd.head(j_corte)

        if dd.empty:
            rows.append({"Equipo": eq, "Pts":0,"PJ":0,"PG":0,"PE":0,"PP":0,"GF":0,"GC":0})
            rachas_dict[eq] = []
            continue

        PJ=PG=PE=PP=GF=GC=PTS=0
        hist = []
        for _, r in dd.iterrows():
            gl, gv = int(r["Goles Local"]), int(r["Goles Visitante"])
            if r["Equipo Local"] == eq:
                gf, gc = gl, gv; rr = _res(gl, gv)
            else:
                gf, gc = gv, gl; rr = _res(gv, gl)

            PJ += 1; GF += gf; GC += gc
            if rr == "G": PG += 1; PTS += 3
            elif rr == "E": PE += 1; PTS += 1
            else: PP += 1
            hist.append(rr)

        rows.append({"Equipo": eq, "Pts": PTS,"PJ": PJ,"PG": PG,"PE": PE,"PP": PP,"GF": GF,"GC": GC})
        rachas_dict[eq] = hist

    df_tab = pd.DataFrame(rows)
    df_tab["DG"] = df_tab["GF"] - df_tab["GC"]
    df_tab = df_tab.sort_values(["Pts","DG","GF"], ascending=[False,False,False]).reset_index(drop=True)
    df_tab.insert(0, "Pos", df_tab.index + 1)

    # Racha viva (últimos iguales consecutivos) sobre esos j_corte partidos
    rachas = []
    for eq in df_tab["Equipo"]:
        h = rachas_dict.get(eq, [])
        if not h:
            rachas.append("—")
        else:
            last = h[-1]; c=1
            for x in reversed(h[:-1]):
                if x==last: c+=1
                else: break
            rachas.append(f"{last}x{c}")
    df_tab["Racha"] = rachas

    # sin Movimiento aquí
    if "Movimiento" in df_tab.columns:
        df_tab = df_tab.drop(columns=["Movimiento"])

    return df_tab



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

    # Objeto del partido (tiene paths preferidos)
    match = get_match_by_label(sel)

    # --- XML para POSESIÓN: intenta TotalValues primero, luego NacSport ---
    XML_TV, _ = infer_paths_for_label(sel)                      # ... - XML TotalValues.xml
    XML_NS    = match["xml_players"] if match else None         # ... - XML NacSport.xml (si existe)
    candidatos = [p for p in (XML_TV, XML_NS) if p and os.path.isfile(p)]

    def _tiene_posesion(xmlp: str) -> bool:
        pm, pr = parse_possession_from_equipo(xmlp)
        return (pm + pr) > 0

    POS_XML = None
    for p in candidatos:               # orden: TotalValues -> NacSport
        if _tiene_posesion(p):
            POS_XML = p
            break
    if POS_XML is None and candidatos: # último recurso
        POS_XML = candidatos[0]

    # 1) Posesión
    pos_m, pos_r = parse_possession_from_equipo(POS_XML) if POS_XML else (0.0, 0.0)

    # 2) Matrix
    MATRIX_PATH = match["matrix_path"] if match else None
    if MATRIX_PATH and os.path.isfile(MATRIX_PATH):
        try:
            mx = compute_from_matrix(MATRIX_PATH)
            FERRO, RIVAL = mx["FERRO"], mx["RIVAL"]
        except Exception as e:
            st.error(f"No pude leer MATRIX '{os.path.basename(MATRIX_PATH)}': {e}")
            FERRO, RIVAL = {}, {}
    else:
        FERRO, RIVAL = {}, {}

    # 2.5) Pases en el último tercio + Pases al área (reusando lógica del Timeline)
    XML_PASS = (match.get("xml_players") if match else None)
    if not (XML_PASS and os.path.isfile(XML_PASS)):
        XML_PASS = POS_XML if (POS_XML and os.path.isfile(POS_XML)) else None

    last_m = last_r = area_m = area_r = 0
    if XML_PASS:
        try:
            jug = parse_instances_jugadores(XML_PASS)                 # ← misma función del Timeline
            last_m, last_r, area_m, area_r = passes_last_third_and_area(jug)  # ← misma función del Timeline
        except Exception as e:
            st.info(f"No pude calcular 'último tercio' / 'al área' desde {os.path.basename(XML_PASS)}: {e}")

    # Inyectar al panel
    FERRO["Posesión %"]          = pos_m
    RIVAL["Posesión %"]          = pos_r
    FERRO["Pases último tercio"] = int(last_m)
    RIVAL["Pases último tercio"] = int(last_r)
    FERRO["Pases al área"]       = int(area_m)
    RIVAL["Pases al área"]       = int(area_r)

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

    # Cargamos XML de jugadores
    df_raw = cargar_datos_nacsport(match["xml_players"]) if match else pd.DataFrame()

    # ⬅️ NUEVO: SOLO jugadores válidos de Ferro (Nombre (Rol)); excluye rival, posesión y no jugado
    df_raw = df_raw[df_raw["jugador"].apply(is_player_code)].copy()

    # Dataset base (jugador, rol, pos_x,pos_y)
    df_xy_all = explode_coords_for_heatmap(df_raw)

    # Listas de nombres/roles presentes en ese partido
    jugadores = sorted(df_xy_all["player_name"].dropna().unique().tolist())
    roles_por_jugador = {
        j: sorted(df_xy_all.loc[df_xy_all["player_name"] == j, "role"].dropna().unique().tolist())
        for j in jugadores
    }

    st.subheader("Opciones de visualización")
    scope = st.radio("Ámbito", ["Equipo entero", "Jugador", "Jugador + Rol"], horizontal=True)

    # Filtro y rotación (la rotación por rol sólo se aplica cuando elegís un rol específico)
    title_suffix = ""
    if scope == "Equipo entero" or not jugadores:
        df_filt = df_xy_all.copy()
        df_rot  = rotate_coords_for_attack_right(df_filt, role=None)
        title_suffix = "Equipo"
        if not jugadores:
            st.info("No se detectaron jugadores en el XML. Se muestra el mapa del equipo.")
    elif scope == "Jugador":
        sel_player = st.selectbox("Jugador", jugadores, index=0)
        df_filt = df_xy_all[df_xy_all["player_name"] == sel_player].copy()
        df_rot  = rotate_coords_for_attack_right(df_filt, role=None)
        title_suffix = f"Jugador: {sel_player}"
        if df_rot.empty:
            st.info("Ese jugador no tiene eventos válidos para el mapa en este partido.")
    else:  # Jugador + Rol
        colp, colr = st.columns([2, 1])
        with colp:
            sel_player = st.selectbox("Jugador", jugadores, index=0)
        roles = roles_por_jugador.get(sel_player, [])
        with colr:
            sel_role = st.selectbox("Rol", roles, index=0 if roles else None)
        if roles:
            df_filt = df_xy_all[(df_xy_all["player_name"] == sel_player) & (df_xy_all["role"] == sel_role)].copy()
            df_rot  = rotate_coords_for_attack_right(df_filt, role=sel_role)  # flip vertical para Ala I/D
            title_suffix = f"Jugador: {sel_player}  |  Rol: {sel_role}"
            if df_rot.empty:
                st.info("No hay eventos para ese jugador en ese rol en este partido.")
        else:
            st.info("Ese jugador no tiene roles registrados en este partido. Se muestra el mapa del equipo.")
            df_filt = df_xy_all.copy()
            df_rot  = rotate_coords_for_attack_right(df_filt, role=None)
            title_suffix = "Equipo"

    # Dibujar
    fig = fig_heatmap(df_rot, f"Mapa de calor — {sel} — {title_suffix}")
    st.pyplot(fig, use_container_width=True)

# =========================
# 🕓 DISTRIBUCIÓN DE MINUTOS
# =========================

elif menu == "🕓 Distribución de minutos":
    # =========================
    # Config / helpers LOCALES (solo afectan a este menú)
    # =========================
    DESC_CANON = [
        "Valla Invicta en cancha",
        "Goles a favor en cancha",
        "Participa en Gol Hecho",
        "Gol Rival en cancha",
        "Involucrado en gol recibido",
    ]
    DESC_CANON_LC = [nlower(x) for x in DESC_CANON]

    def _prep_minutes_table(df: pd.DataFrame, include_role: bool=False) -> pd.DataFrame:
        """
        Tabla 'Minutos':
          - 'minutos' (MM:SS) + 'nT'
          - quita 'segundos' y 'minutos' numérica
          - garantiza columnas de descriptores (0 si faltan)
        """
        base_cols = (["nombre","rol"] if include_role else ["nombre"]) + ["minutos","nT"] + DESC_CANON
        if df is None or df.empty:
            return pd.DataFrame(columns=base_cols)

        d = df.copy()
        for c in DESC_CANON:
            if c not in d.columns:
                d[c] = 0
        # quitar numéricas internas
        for c in ["segundos","minutos"]:
            if c in d.columns:
                d = d.drop(columns=[c])
        # renombres amables
        if "mmss" in d.columns:      d = d.rename(columns={"mmss": "minutos"})
        if "n_tramos" in d.columns:  d = d.rename(columns={"n_tramos": "nT"})
        keep = [c for c in base_cols if c in d.columns]
        d = d.loc[:, keep]
        d = d.loc[:, ~d.columns.duplicated(keep="first")]
        d[DESC_CANON] = d[DESC_CANON].fillna(0).astype(int)
        return d

    def _prep_impact_table(df: pd.DataFrame, include_role: bool, total_secs_scope: int) -> pd.DataFrame:
        """
        Tabla 'Impacto':
          - columnas base de minutos (MM:SS) + nT + 5 descriptores
          - Impacto +, Impacto −, Impacto neto (escala seg/2400)
          - % minutos (sobre total alcance)
          - % CS/nT, % PF/GF_on, % IA/GA_on
        """
        if df is None or df.empty:
            cols = (["nombre","rol"] if include_role else ["nombre"]) + [
                "minutos","nT","Impacto +","Impacto −","Impacto neto",
                "% minutos","% CS/nT","% PF/GF_on","% IA/GA_on"
            ] + DESC_CANON
            return pd.DataFrame(columns=cols)

        d = df.copy()
        base = _prep_minutes_table(d, include_role=include_role)

        # asegurar columnas necesarias
        for c in ["segundos","n_tramos","Valla Invicta en cancha",
                  "Goles a favor en cancha","Participa en Gol Hecho",
                  "Gol Rival en cancha","Involucrado en gol recibido"]:
            if c not in d.columns:
                d[c] = 0

        # % minutos (sobre total del alcance)
        tot = float(total_secs_scope) if total_secs_scope else float(d["segundos"].sum())
        pct_mins = (d["segundos"] / tot * 100.0) if tot else 0.0

        # % CS vs nT
        nT = d.get("n_tramos", d.get("nT", 0))
        cs = d["Valla Invicta en cancha"]
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_cs_nt = np.where(nT.to_numpy()>0, cs.to_numpy()/nT.to_numpy()*100.0, 0.0)

        # % PF vs GF_on
        pf = d["Participa en Gol Hecho"]; gf = d["Goles a favor en cancha"]
        pct_pf_gf = np.where(gf.to_numpy()>0, pf.to_numpy()/gf.to_numpy()*100.0, 0.0)

        # % IA vs GA_on
        ia = d["Involucrado en gol recibido"]; ga = d["Gol Rival en cancha"]
        pct_ia_ga = np.where(ga.to_numpy()>0, ia.to_numpy()/ga.to_numpy()*100.0, 0.0)

        view = base.copy()
        view["Impacto +"]     = d["Impacto +"].round(3)
        view["Impacto −"]     = d["Impacto −"].round(3)
        view["Impacto neto"]  = d["Impacto neto"].round(3)
        view["% minutos"]     = np.round(pct_mins, 1)
        view["% CS/nT"]       = np.round(pct_cs_nt, 1)
        view["% PF/GF_on"]    = np.round(pct_pf_gf, 1)
        view["% IA/GA_on"]    = np.round(pct_ia_ga, 1)

        cols_front = (["nombre","rol"] if include_role else ["nombre"]) + [
            "minutos","nT","Impacto +","Impacto −","Impacto neto",
            "% minutos","% CS/nT","% PF/GF_on","% IA/GA_on"
        ]
        view = view[[c for c in cols_front if c in view.columns] + [c for c in DESC_CANON if c in view.columns]]
        return view

    def _fig_bar_minutos(labels, secs_list, ntramos_list, title,
                         sort_desc=True, label_place="inside"):
        """
        Barras de minutos ordenadas (desc por defecto) y etiquetas consistentes.
        label_place: "inside" o "outside" (todas iguales, blancas).
        """
        # ordenar por minutos (desc)
        rows = list(zip(labels, secs_list, ntramos_list))
        rows.sort(key=lambda t: t[1], reverse=bool(sort_desc))
        labels, secs_list, ntramos_list = (list(x) for x in zip(*rows)) if rows else ([], [], [])

        mins = np.array(secs_list, dtype=float) / 60.0
        H = max(3.8, 0.48*len(labels))
        fig, ax = plt.subplots(figsize=(10, H))
        bars = ax.barh(labels, mins, alpha=0.9)
        ax.invert_yaxis()  # mayor arriba
        ax.set_xlabel("Minutos")
        ax.set_title(title)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        vmax = float(mins.max() if len(mins) else 1.0)
        ax.set_xlim(0, vmax*1.12 + 0.4)

        # etiquetas consistentes
        try:
            import matplotlib.patheffects as pe
            pe_stroke = [pe.withStroke(linewidth=2.2, foreground=bg_green)]
        except Exception:
            pe_stroke = None

        for b, secs, nt in zip(bars, secs_list, ntramos_list):
            txt = f"{_format_mmss(secs)} ({nt})"
            x = b.get_width()
            y = b.get_y() + b.get_height()/2
            if label_place == "inside":
                ax.text(x - 0.08, y, txt, va="center", ha="right",
                        fontsize=10, color="white", fontweight="normal")
            else:  # outside
                kwargs = dict(fontsize=10, color="white", fontweight="normal")
                if pe_stroke: kwargs["path_effects"] = pe_stroke
                ax.text(x + 0.10, y, txt, va="center", ha="left", **kwargs)

        plt.tight_layout()
        return fig

    def _tv_load_presencias(xml_path: str, partido_label: str) -> pd.DataFrame:
        """
        Lee SOLO XML TotalValues (no NacSport) y devuelve instancias válidas para minutos:
          - Jugador (Rol) válido (is_player_code)
          - labels vacíos  O  alguno ∈ DESC_CANON
        """
        cols = ["nombre","rol","start_s","end_s","dur_s","labels_lc","partido"]
        if not xml_path or not os.path.isfile(xml_path):
            return pd.DataFrame(columns=cols)

        root = ET.parse(xml_path).getroot()
        rows = []
        for inst in root.findall(".//instance"):
            code = inst.findtext("code") or ""
            if not is_player_code(code):
                continue
            labels_txt = [(lbl.findtext("text") or "").strip() for lbl in inst.findall("label")]
            labels_lc  = [nlower(t) for t in labels_txt if t.strip()!=""]
            aceptar = (len(labels_lc) == 0) or any(lc in DESC_CANON_LC for lc in labels_lc)
            if not aceptar:
                continue
            m = _NAME_ROLE_RE.match(code)
            if not m:
                continue
            nombre = m.group(1).strip()
            rol    = m.group(2).strip()
            stt = inst.findtext("start"); enn = inst.findtext("end")
            try:
                s = float(stt) if stt is not None else None
                e = float(enn) if enn is not None else None
            except Exception:
                s, e = None, None
            if s is None or e is None or e <= s:
                continue

            rows.append({
                "nombre": nombre, "rol": rol,
                "start_s": s, "end_s": e, "dur_s": e - s,
                "labels_lc": labels_lc, "partido": partido_label,
            })
        return pd.DataFrame(rows, columns=cols)

    def _load_all_tv_presencias() -> pd.DataFrame:
        """Carga y concatena TODAS las presencias de todos los partidos (solo TotalValues)."""
        matches_obj = discover_matches()
        all_rows = []
        for m in matches_obj:
            label = m["label"]
            xml_tv, _mx = infer_paths_for_label(label)  # fuerza TotalValues
            if xml_tv and os.path.isfile(xml_tv):
                dfm = _tv_load_presencias(xml_tv, partido_label=label)
                if not dfm.empty:
                    all_rows.append(dfm)
        if not all_rows:
            return pd.DataFrame(columns=["nombre","rol","start_s","end_s","dur_s","labels_lc","partido"])
        return pd.concat(all_rows, ignore_index=True)

    def _descriptor_counts(df_pres: pd.DataFrame):
        """Conteos de descriptores por (nombre, rol) y por (nombre) total."""
        if df_pres is None or df_pres.empty:
            por_rol = pd.DataFrame(columns=["nombre","rol"] + DESC_CANON)
            por_jug = pd.DataFrame(columns=["nombre"] + DESC_CANON)
            return por_rol, por_jug

        tmp = df_pres.explode("labels_lc", ignore_index=True)
        tmp = tmp.dropna(subset=["labels_lc"])
        tmp = tmp[tmp["labels_lc"].isin(DESC_CANON_LC)].copy()
        if tmp.empty:
            por_rol = df_pres[["nombre","rol"]].drop_duplicates().copy()
            for c in DESC_CANON: por_rol[c] = 0
            por_jug = df_pres[["nombre"]].drop_duplicates().copy()
            for c in DESC_CANON: por_jug[c] = 0
            return por_rol, por_jug

        def _canon_of(lc: str) -> str:
            for c, cl in zip(DESC_CANON, DESC_CANON_LC):
                if lc == cl: return c
            return lc

        tmp["desc_canon"] = tmp["labels_lc"].map(_canon_of)

        por_rol = (tmp.groupby(["nombre","rol","desc_canon"])
                      .size().unstack("desc_canon", fill_value=0).reset_index())
        for c in DESC_CANON:
            if c not in por_rol.columns:
                por_rol[c] = 0
        por_rol = por_rol[["nombre","rol"] + DESC_CANON]

        por_jug = (tmp.groupby(["nombre","desc_canon"])
                      .size().unstack("desc_canon", fill_value=0).reset_index())
        for c in DESC_CANON:
            if c not in por_jug.columns:
                por_jug[c] = 0
        por_jug = por_jug[["nombre"] + DESC_CANON]
        return por_rol, por_jug

    def _agg_minutes(df_pres: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Minutos y nº de tramos con merge de intervalos por partido y luego suma.
        mode ∈ {"jug_total","jug_rol"}.
        """
        if df_pres is None or df_pres.empty:
            if mode == "jug_rol":
                return pd.DataFrame(columns=["nombre","rol","segundos","mmss","minutos","n_tramos"])
            else:
                return pd.DataFrame(columns=["nombre","segundos","mmss","minutos","n_tramos"])

        if mode == "jug_rol":
            base_keys  = ["nombre","rol","partido"]; final_keys = ["nombre","rol"]
        else:
            base_keys  = ["nombre","partido"];       final_keys = ["nombre"]

        rows = []
        for keys, g in df_pres.groupby(base_keys, dropna=False):
            intervals = list(zip(g["start_s"], g["end_s"]))
            merged = _merge_intervals(intervals)
            secs = int(round(sum((e - s) for s, e in merged)))
            rows.append({**{k:v for k,v in zip(base_keys, keys)}, "segundos": secs, "n_tramos": len(merged)})

        base_df = pd.DataFrame(rows)
        out = (base_df.groupby(final_keys, as_index=False)
                         .agg(segundos=("segundos","sum"), n_tramos=("n_tramos","sum")))
        out["mmss"]    = out["segundos"].apply(_format_mmss)
        out["minutos"] = (out["segundos"] / 60.0).round(2)
        out = out.sort_values(final_keys + ["segundos"],
                              ascending=[True]*len(final_keys) + [False])
        return out

    def _total_scope_seconds(df_pres: pd.DataFrame) -> int:
        """Total de segundos del alcance usando los intervalos del/los 'Arq' por partido."""
        if df_pres is None or df_pres.empty:
            return 0
        roles_lc = df_pres["rol"].astype(str).str.lower()
        is_gk = roles_lc.isin({"arq","arquero","gk"})
        if not is_gk.any():
            # fallback: duración aprox por partido
            sec = 0.0
            for _, g in df_pres.groupby("partido", dropna=False):
                sec += float(g["end_s"].max())
            return int(round(sec))
        tot = 0.0
        for _, g in df_pres[is_gk].groupby("partido", dropna=False):
            merged = _merge_intervals(list(zip(g["start_s"], g["end_s"])))
            tot += sum(e - s for s, e in merged)
        return int(round(tot))

    def _add_impacts(d: pd.DataFrame, mode: str, gk_names: set[str] | None = None) -> pd.DataFrame:
        """
        Añade Impacto + / − / neto con escala a 40' = segundos/2400.
        - Campo:  P+=0.60*PF + 0.30*GF_on + 0.10*CS
                  N−=0.60*IA + 0.30*GA_on + 0.10*noCS
        - Arq  :  P+=0.80*CS + 0.10*GF_on + 0.10*PF
                  N−=0.80*noCS + 0.10*GA_on + 0.10*IA
        """
        if d is None or d.empty:
            return d

        out = d.copy()
        for c in ["Valla Invicta en cancha","Goles a favor en cancha","Participa en Gol Hecho",
                  "Gol Rival en cancha","Involucrado en gol recibido","n_tramos","segundos"]:
            if c not in out.columns: out[c] = 0

        out["noCS"]   = np.maximum(0, out["n_tramos"] - out["Valla Invicta en cancha"])
        out["scale"]  = out["segundos"] / 2400.0  # escala pedida

        def _is_gk_row(row) -> bool:
            if mode == "jug_rol":
                return str(row.get("rol","")).strip().lower() in {"arq","arquero","gk"}
            return row["nombre"] in (gk_names or set())

        P_vals, N_vals = [], []
        for _, r in out.iterrows():
            gk = _is_gk_row(r)
            if gk:
                P_raw = 0.80*r["Valla Invicta en cancha"] + 0.10*r["Goles a favor en cancha"] + 0.10*r["Participa en Gol Hecho"]
                N_raw = 0.80*r["noCS"] + 0.10*r["Gol Rival en cancha"] + 0.10*r["Involucrado en gol recibido"]
            else:
                P_raw = 0.60*r["Participa en Gol Hecho"] + 0.30*r["Goles a favor en cancha"] + 0.10*r["Valla Invicta en cancha"]
                N_raw = 0.60*r["Involucrado en gol recibido"] + 0.30*r["Gol Rival en cancha"] + 0.10*r["noCS"]
            P_vals.append(P_raw * r["scale"])
            N_vals.append(N_raw * r["scale"])

        out["Impacto +"]    = np.round(P_vals, 3)
        out["Impacto −"]    = np.round(N_vals, 3)
        out["Impacto neto"] = np.round(out["Impacto +"] - out["Impacto −"], 3)
        return out

    # =========================
    # UI — Alcance & Panel
    # =========================
    data_scope = st.radio("Alcance", ["Partido", "Todos los partidos"], horizontal=True)
    panel = st.selectbox("Panel", ["Minutos", "Impacto"], index=0)

    # ---- Carga de presencias según alcance
    if data_scope == "Partido":
        matches = discover_matches()
        if not matches:
            st.warning("No encontré partidos en data/minutos.")
            st.stop()
        sel = st.selectbox("Elegí partido", [m["label"] for m in matches], index=0)

        XML_TV, _mx = infer_paths_for_label(sel)  # fuerza TotalValues
        if not XML_TV or not os.path.isfile(XML_TV):
            st.error("Para este módulo necesito el XML TotalValues del partido.")
            st.stop()
        df_pres = _tv_load_presencias(XML_TV, partido_label=sel)
    else:
        df_pres = _load_all_tv_presencias()
        if df_pres.empty:
            st.warning("No encontré XML TotalValues válidos para acumular.")
            st.stop()

    # ---- Minutos/Tramos + Descriptores
    dj_total = _agg_minutes(df_pres, mode="jug_total")     # por jugador
    dr_total = _agg_minutes(df_pres, mode="jug_rol")       # por jugador&rol
    desc_por_rol, desc_por_jug = _descriptor_counts(df_pres)

    # ---- Merge minutos + descriptores
    dj_merged = pd.merge(dj_total, desc_por_jug, on="nombre", how="left")
    dj_merged[DESC_CANON] = dj_merged[DESC_CANON].fillna(0).astype(int)
    dr_merged = pd.merge(dr_total, desc_por_rol, on=["nombre","rol"], how="left")
    dr_merged[DESC_CANON] = dr_merged[DESC_CANON].fillna(0).astype(int)

    # ---- Set de arqueros (para pesos en jugador total)
    gk_names = set(df_pres.loc[df_pres["rol"].astype(str).str.lower().isin({"arq","arquero","gk"}), "nombre"].unique().tolist())

    # ---- Total de segundos del alcance (para % minutos)
    total_secs_scope = _total_scope_seconds(df_pres)

    # =========================
    # Panel: MINUTOS
    # =========================
    if panel == "Minutos":
        scope = st.radio("Ver:", ["Jugador total", "Por rol"], horizontal=True)
        etiqueta_pos = st.radio("Etiquetas en barras", ["Dentro", "Fuera"], index=0, horizontal=True)
        label_place = "inside" if etiqueta_pos == "Dentro" else "outside"

        if scope == "Jugador total":
            st.subheader("⏱️ Minutos totales por jugador (con descriptores)")
            view = _prep_minutes_table(dj_merged, include_role=False)
            show_full_table(view)

            if not dj_merged.empty:
                fig = _fig_bar_minutos(
                    labels=dj_merged["nombre"].tolist(),
                    secs_list=dj_merged["segundos"].tolist(),
                    ntramos_list=dj_merged["n_tramos"].tolist(),
                    title=("Minutos totales por jugador" if data_scope=="Todos los partidos"
                           else "Minutos totales por jugador (partido seleccionado)"),
                    sort_desc=True,
                    label_place=label_place
                )
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Sin datos válidos.")

        else:
            roles_presentes = sorted([r for r in dr_merged["rol"].dropna().unique().tolist()])
            if not roles_presentes:
                st.info("No hay roles registrados en el alcance seleccionado.")
                st.stop()

            sel_rol = st.selectbox("Rol", roles_presentes, index=0)
            drol = dr_merged[dr_merged["rol"] == sel_rol].copy()

            st.subheader(f"⏱️ Jugadores en rol: {sel_rol}")
            view = _prep_minutes_table(drol, include_role=True)
            show_full_table(view)

            if not drol.empty:
                fig = _fig_bar_minutos(
                    labels=(drol["nombre"] + " (" + drol["rol"] + ")").tolist(),
                    secs_list=drol["segundos"].tolist(),
                    ntramos_list=drol["n_tramos"].tolist(),
                    title=(f"Minutos en rol {sel_rol} — acumulado" if data_scope=="Todos los partidos"
                           else f"Minutos en rol {sel_rol} — partido seleccionado"),
                    sort_desc=True,
                    label_place=label_place
                )
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Ese rol no tiene jugadores en el alcance seleccionado.")

    # =========================
    # Panel: IMPACTO (+/−/neto) con escala a 40' = seg/2400
    # =========================
    else:
        scope = st.radio("Ver:", ["Jugador total", "Por rol"], horizontal=True)

        if scope == "Jugador total":
            dj_imp = _add_impacts(dj_merged, mode="jug_total", gk_names=gk_names)
            view = _prep_impact_table(dj_imp, include_role=False, total_secs_scope=total_secs_scope)

            tabP, tabN, tabNet = st.tabs(["Orden: Impacto + ↓", "Orden: Impacto − ↓", "Orden: Impacto neto ↓"])
            with tabP:
                show_full_table(view.sort_values("Impacto +", ascending=False).reset_index(drop=True))
            with tabN:
                show_full_table(view.sort_values("Impacto −", ascending=False).reset_index(drop=True))
            with tabNet:
                show_full_table(view.sort_values("Impacto neto", ascending=False).reset_index(drop=True))

        else:
            roles_presentes = sorted([r for r in dr_merged["rol"].dropna().unique().tolist()])
            if not roles_presentes:
                st.info("No hay roles registrados en el alcance seleccionado.")
                st.stop()
            sel_rol = st.selectbox("Rol", roles_presentes, index=0)

            drol = dr_merged[dr_merged["rol"] == sel_rol].copy()
            drol_imp = _add_impacts(drol, mode="jug_rol", gk_names=None)
            view = _prep_impact_table(drol_imp, include_role=True, total_secs_scope=total_secs_scope)

            tabP, tabN, tabNet = st.tabs([f"{sel_rol} — Impacto + ↓", f"{sel_rol} — Impacto − ↓", f"{sel_rol} — Neto ↓"])
            with tabP:
                show_full_table(view.sort_values("Impacto +", ascending=False).reset_index(drop=True))
            with tabN:
                show_full_table(view.sort_values("Impacto −", ascending=False).reset_index(drop=True))
            with tabNet:
                show_full_table(view.sort_values("Impacto neto", ascending=False).reset_index(drop=True))

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
# 📋 TABLA & RESULTADOS (REEMPLAZAR TODO EL BLOQUE)
# =========================
if menu == "🏆 Tabla & Resultados":
    import requests, pandas as pd, numpy as np
    from datetime import datetime, date
    import time
    import matplotlib.pyplot as plt

    st.subheader("📋 Tabla & Resultados")

    # ---------- Helpers genéricos ----------
    def _autoh(st_df, row_px=38):
        """Altura automática para mostrar la tabla completa sin scroll."""
        n = int(getattr(st_df, "shape", (0,0))[0])
        return max(140, int(row_px * (n + 1)))

    def show_full_table(df):
        st.dataframe(df, use_container_width=True, height=_autoh(df))

    # --- Parámetros de la competencia (los tuyos) ---
    URL_TABLA   = "https://api.weball.me/public/tournament/176/phase/150/group/613/clasification?instanceUUID=2d260df1-7986-49fd-95a2-fcb046e7a4fb"
    URL_MATCHES = "https://api.weball.me/public/tournament/176/phase/150/matches?instanceUUID=2d260df1-7986-49fd-95a2-fcb046e7a4fb"
    HEADERS     = {"Content-Type": "application/json"}
    CATEG_FILTRO = "2016 PROMOCIONALES"
    EXCLUIR_LIBRE = True

    # ---------- Fetchers cacheados (sin tocar tu lógica) ----------
    @st.cache_data(ttl=300, show_spinner=False)
    def _safe_get(url, max_tries=3, sleep_s=0.8):
        last_err = None
        for _ in range(max_tries):
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e; time.sleep(sleep_s)
        raise last_err

    @st.cache_data(ttl=300, show_spinner=False)
    def _safe_post(url, payload, max_tries=3, sleep_s=0.8):
        last_err = None
        for _ in range(max_tries):
            try:
                r = requests.post(url, headers=HEADERS, json=payload, timeout=20)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e; time.sleep(sleep_s)
        raise last_err

    @st.cache_data(ttl=300)
    def fetch_tabla():
        data = _safe_get(URL_TABLA)
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

                if EXCLUIR_LIBRE and (local.upper() == "LIBRE" or (visitante or "").upper() == "LIBRE"):
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

        # DESPUÉS (normaliza a naive, sin tz)
        df["Fecha Técnica"] = (
            pd.to_datetime(df["Fecha Técnica"], utc=True, errors="coerce")
              .dt.tz_convert(None)
        )


        fin_mask = df["Estado"].str.lower().eq("finalizado")
        df_res = df[fin_mask].copy().sort_values("Fecha Técnica")
        df_fix = df[~df["Estado"].str.lower().isin(["finalizado","cancelado"])].copy()

        return df, df_res, df_fix

    # ---------- Utilidades de “Tabla fecha a fecha” ----------
    def _resultado(gl, gv):
        if gl > gv: return "G"
        if gl < gv: return "P"
        return "E"

    def tabla_a_fecha(df_res, corte_ts):
        """Tabla a una fecha de corte."""
        d = df_res[df_res["Fecha Técnica"] <= corte_ts].copy()
        if d.empty:
            return pd.DataFrame(columns=["Pos","Equipo","Pts","PJ","PG","PE","PP","GF","GC","DG","Movimiento","Racha"])
        rows = {}
        equipos = pd.unique(pd.concat([d["Equipo Local"], d["Equipo Visitante"]]))
        for eq in equipos:
            rows[eq] = {"Equipo": eq, "Pts":0,"PJ":0,"PG":0,"PE":0,"PP":0,"GF":0,"GC":0}

        for _, r in d.iterrows():
            gl, gv = int(r["Goles Local"]), int(r["Goles Visitante"])
            el, ev = r["Equipo Local"], r["Equipo Visitante"]
            res_l = _resultado(gl, gv); res_v = _resultado(gv, gl)
            # Local
            rows[el]["PJ"] += 1; rows[el]["GF"] += gl; rows[el]["GC"] += gv
            if res_l=="G": rows[el]["PG"]+=1; rows[el]["Pts"]+=3
            elif res_l=="E": rows[el]["PE"]+=1; rows[el]["Pts"]+=1
            else: rows[el]["PP"]+=1
            # Visitante
            rows[ev]["PJ"] += 1; rows[ev]["GF"] += gv; rows[ev]["GC"] += gl
            if res_v=="G": rows[ev]["PG"]+=1; rows[ev]["Pts"]+=3
            elif res_v=="E": rows[ev]["PE"]+=1; rows[ev]["Pts"]+=1
            else: rows[ev]["PP"]+=1

        df_tab = pd.DataFrame(rows.values())
        df_tab["DG"] = df_tab["GF"] - df_tab["GC"]
        df_tab = df_tab.sort_values(["Pts","DG","GF"], ascending=[False,False,False]).reset_index(drop=True)
        df_tab.insert(0, "Pos", df_tab.index + 1)

        # Movimiento vs snapshot anterior
        prev_dates = d["Fecha Técnica"].unique()
        prev_dates = prev_dates[prev_dates < corte_ts]
        if prev_dates.size:
            prev = tabla_a_fecha(df_res, prev_dates.max()).set_index("Equipo")["Pos"]
            mov = []
            for _, row in df_tab.iterrows():
                eq, pos = row["Equipo"], int(row["Pos"])
                pprev = int(prev.get(eq, pos))
                delta = pprev - pos
                if   delta >  0: mov.append(f"↑ +{delta}")
                elif delta <  0: mov.append(f"↓ {delta}")
                else:            mov.append("= 0")
            df_tab["Movimiento"] = mov
        else:
            df_tab["Movimiento"] = "= 0"

        # Racha viva (G/E/P consecutivos al final)
        rachas = []
        for eq in df_tab["Equipo"]:
            h = []
            for _, r in d[(d["Equipo Local"]==eq) | (d["Equipo Visitante"]==eq)].sort_values("Fecha Técnica").iterrows():
                if r["Equipo Local"]==eq: h.append(_resultado(int(r["Goles Local"]), int(r["Goles Visitante"])))
                else:                     h.append(_resultado(int(r["Goles Visitante"]), int(r["Goles Local"])))
            if not h:
                rachas.append("—")
            else:
                last = h[-1]; c=1
                for x in reversed(h[:-1]):
                    if x==last: c+=1
                    else: break
                tag = {"G":"G","E":"E","P":"P"}[last] + f"x{c}"
                rachas.append(tag)
        df_tab["Racha"] = rachas
        return df_tab

    def build_elo(df_res, k=24, base=1000):
        if df_res.empty:
            return pd.DataFrame(columns=["Fecha","Equipo","ELO"])
        teams = pd.unique(pd.concat([df_res["Equipo Local"], df_res["Equipo Visitante"]])).tolist()
        rating = {t: base for t in teams}
        rows = []
        for _, r in df_res.sort_values("Fecha Técnica").iterrows():
            el, ev = r["Equipo Local"], r["Equipo Visitante"]
            gl, gv = int(r["Goles Local"]), int(r["Goles Visitante"])
            Ra, Rb = rating[el], rating[ev]
            Ea = 1.0 / (1 + 10 ** ((Rb - Ra) / 400))
            Eb = 1.0 - Ea
            Sa = 1.0 if gl > gv else (0.5 if gl==gv else 0.0)
            Sb = 1.0 - Sa
            rating[el] = Ra + k * (Sa - Ea)
            rating[ev] = Rb + k * (Sb - Eb)
            rows.append({"Fecha": r["Fecha Técnica"], "Equipo": el, "ELO": rating[el]})
            rows.append({"Fecha": r["Fecha Técnica"], "Equipo": ev, "ELO": rating[ev]})
        elo = pd.DataFrame(rows)
        elo["Fecha"] = pd.to_datetime(elo["Fecha"])
        return elo

    def plot_elo_suave(elo_df, corte_ts):
        if elo_df.empty:
            st.info("Sin datos de ELO para graficar."); return
        d = elo_df[elo_df["Fecha"] <= corte_ts]
        if d.empty:
            st.info("No hay ELO antes del corte seleccionado."); return

        fig, ax = plt.subplots(figsize=(10, 6))
        teams = sorted(d["Equipo"].unique())
        cmap = plt.get_cmap('tab20', len(teams))
        colors = {t: cmap(i) for i,t in enumerate(teams)}

        tips = []  # (team, x_last, y_last)
        for t in teams:
            s = (d[d["Equipo"]==t]
                 .sort_values("Fecha")
                 .set_index("Fecha")["ELO"]
                 .asfreq("D")
                 .interpolate(method="time")
                 .rolling(3, min_periods=1).mean())  # suavizado suave
            ax.plot(s.index, s.values, lw=1.9, color=colors[t])
            tips.append((t, s.index[-1], float(s.values[-1])))

        # Etiquetas en la punta, separadas
        tips.sort(key=lambda x: x[2])
        offs = np.linspace(-8, 8, num=len(tips))
        xmax = max(x for _,x,_ in tips)
        for off, (t, x, y) in zip(offs, tips):
            ax.text(x + pd.Timedelta(days=3), y + off, t, fontsize=9, va="center", color=colors[t])
            ax.plot([x, x + pd.Timedelta(days=3)], [y, y + off], lw=0.8, color=colors[t], alpha=0.6)

        ax.set_xlim(d["Fecha"].min(), xmax + pd.Timedelta(days=20))
        ax.set_ylabel("Índice ELO")
        ax.grid(True, ls="--", alpha=0.3)
        st.pyplot(fig)

    def build_wdl(df_res):
        if df_res.empty:
            return pd.DataFrame()
        rows = []
        for _, r in df_res.iterrows():
            gl, gv = int(r["Goles Local"]), int(r["Goles Visitante"])
            for eq, gf, gc in [(r["Equipo Local"], gl, gv), (r["Equipo Visitante"], gv, gl)]:
                res = "W" if gf>gc else ("D" if gf==gc else "L")
                rows.append({"Fecha": r["Fecha Técnica"], "Equipo": eq, "GF": gf, "GC": gc, "R": res,
                             "Pts": 3 if res=="W" else (1 if res=="D" else 0)})
        out = pd.DataFrame(rows).sort_values("Fecha")
        return out

    # ---------- Datos base ----------
    df_tabla = fetch_tabla()
    df_all, df_res, df_fix = fetch_partidos()

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Tabla actual", "📜 Resultados", "📅 Próximos", "📊 Tabla fecha a fecha"])

    # --- TAB 1: TABLA ACTUAL (completa) ---
    with tab1:
        show_full_table(df_tabla)

    # --- TAB 2: RESULTADOS (filtros equipo + fecha) ---
    with tab2:
        d = df_res.copy()
    
        if d.empty:
            st.info("Sin resultados finalizados.")
        else:
            # Asegurar datetime naive (sin tz) para comparaciones
            d["Fecha Técnica"] = pd.to_datetime(d["Fecha Técnica"], errors="coerce", utc=True).dt.tz_convert(None)
    
            equipos = sorted(pd.unique(pd.concat([d["Equipo Local"], d["Equipo Visitante"]]).dropna()))
            col1, col2 = st.columns([2, 2])
            with col1:
                sel_eq = st.multiselect("Equipo(s)", equipos)
            with col2:
                fmin, fmax = d["Fecha Técnica"].min().date(), d["Fecha Técnica"].max().date()
                rango = st.date_input("Rango de fechas", (fmin, fmax))
    
            # Filtro
            mask = pd.Series(True, index=d.index)
    
            if sel_eq:
                mask &= (d["Equipo Local"].isin(sel_eq) | d["Equipo Visitante"].isin(sel_eq))
    
            # --- ESTA ES LA PARTE QUE TE MARCABA EL ERROR (ojo con la sangría) ---
            if isinstance(rango, (list, tuple)) and len(rango) == 2:
                d1 = pd.Timestamp(rango[0])  # inicio del día
                d2 = pd.Timestamp(rango[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)  # fin del día
                mask &= d["Fecha Técnica"].between(d1, d2)
    
            out = d.loc[mask].sort_values("Fecha Técnica")
            show_full_table(out)

     # --- TAB 3: PRÓXIMOS (solo futuros + filtro equipo) ---
    with tab3:
        d = df_fix.copy()
    
        # 1) Normalizá la columna a datetime "naive" (sin timezone)
        d["Fecha Técnica"] = pd.to_datetime(d["Fecha Técnica"], errors="coerce", utc=True).dt.tz_convert(None)
    
        # 2) >>> ESTE ES EL CORTE A FUTURO <<<
        #    "today" a las 00:00 (hoy), y nos quedamos solo con fechas >= hoy
        today = pd.Timestamp.now().normalize()
        d = d[d["Fecha Técnica"] >= today]
    
        if d.empty:
            st.info("Sin próximos programados.")
        else:
            # filtros por equipo (opcional)
            equipos = sorted(pd.unique(pd.concat([d["Equipo Local"], d["Equipo Visitante"]]).dropna()))
            sel_eq = st.multiselect("Equipo(s)", equipos)
            if sel_eq:
                d = d[(d["Equipo Local"].isin(sel_eq)) | (d["Equipo Visitante"].isin(sel_eq))]
    
            d = d.sort_values("Fecha Técnica")
            # mostrar tabla completa (sin scroll)
            st.dataframe(d, use_container_width=True, height=max(140, 38 * (len(d) + 1)))

    # --- TAB 4: TABLA FECHA A FECHA + ELO + W/D/L ---
    with tab4:
        if df_res.empty:
            st.info("Aún no hay partidos finalizados para construir la tabla por fecha.")
        else:
            # --- Slider por JORNADA (máximo uniforme: todos con la misma cantidad de PJ) ---
            df_res_j, _ = _build_jornada_index(df_res)
            d_ok = df_res_j.dropna(subset=["JornadaN","Goles Local","Goles Visitante"]).copy()
            
            # Máximo de jornadas observado (y forzá 15 si tu torneo tiene 15)
            J_MAX = pd.to_numeric(d_ok["JornadaN"], errors="coerce").dropna()
            J_MAX = int(J_MAX.max()) if not J_MAX.empty else 1
            J_MAX = max(15, J_MAX)  # dejalo en 15 si ese es tu total real
            
            j_corte = st.slider("Corte por fecha (Jornada)", min_value=1, max_value=J_MAX, value=J_MAX, step=1)
            st.caption(f"Mostrando hasta la jornada {j_corte}.")

            # 2) TABLA a la jornada (PJ parejos) y sin recorte por fecha real
            df_fecha = tabla_a_jornada(df_res, j_corte)
            # Ocultar columna 'Movimiento' si está
            df_fecha = df_fecha.drop(columns=["Movimiento"], errors="ignore")
            show_full_table(df_fecha)

            # 3) ELO por jornada (mismo corte), con multiselect
            elo_pivot = _compute_elo_by_jornada(df_res_j[df_res_j["JornadaN"].le(j_corte)])
            equipos_all = list(elo_pivot.columns)
            sel_equipos = st.multiselect("Equipos a mostrar en el ELO", options=equipos_all, default=equipos_all)
            fig_elo = plot_elo_por_jornada(elo_pivot, sel_equipos, j_corte)
            st.pyplot(fig_elo, use_container_width=True)
    
            # 4) W/D/L por jornada (dos bandas)
            # wdl_jornada_df = build_wdl_por_jornada(df_res_j[df_res_j["JornadaN"].le(j_corte)])
            src_wdl = df_res_j[df_res_j["JornadaN"].le(j_corte)]
            wdl_jornada_df = (build_wdl_jornada(src_wdl) if "build_wdl_jornada" in globals()
                              else build_wdl_por_jornada(src_wdl))

            eqs = sorted(wdl_jornada_df["Equipo"].unique())
            c1, c2 = st.columns(2)
            with c1:
                eq1 = st.selectbox("Equipo A", eqs, index=3)
            with c2:
                eq2 = st.selectbox("Equipo B (opcional)", ["(ninguno)"] + eqs, index=0)
            
            fig_wdl = plot_wdl_por_jornada(
                wdl_jornada_df,
                eq1,
                None if eq2 == "(ninguno)" else eq2,
                j_corte
            )
            st.pyplot(fig_wdl, use_container_width=True)


