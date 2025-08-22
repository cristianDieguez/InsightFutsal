# app.py — InsightFutsal (Estadísticas de partido + paths corregidos)

import os, re, unicodedata, math, glob
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")
st.title("InsightFutsal")

DATA_MINUTOS = "data/minutos"   # XML TotalValues
DATA_MATRIX  = "data/matrix"    # Matrix.xlsx / Matrix.csv
BADGE_DIR    = "images/equipos" # ferro.png, <rival>.png
BANNER_DIR   = "images/banner"  # opcional (no obligatorio)

# --- estilo panel (idéntico a tu notebook) ---
bg_green   = "#006633"
text_w     = "#FFFFFF"
bar_white  = "#FFFFFF"
bar_rival  = "#E6EEF2"
bar_rail   = "#0F5E29"
star_c     = "#FFD54A"
loser_alpha = 0.35
orange_win = "#FF8F00"

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

# Orden y tipos (exacto a tu panel)
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
# HELPERS
# =========================
def ntext(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.strip()
def nlower(s): return ntext(s).lower()

# ---- LISTADO: Fecha N° - Rival (desde XML en data/minutos) ----
def list_matches() -> List[str]:
    """
    Busca archivos:
      data/minutos/Fecha N° - Rival - XML TotalValues.xml
    y devuelve labels 'Fecha N° - Rival'
    """
    if not os.path.isdir(DATA_MINUTOS): return []
    pats = glob.glob(os.path.join(DATA_MINUTOS, "Fecha * - * - XML TotalValues.xml"))
    labels = []
    rx = re.compile(r"^Fecha\s*([\d]+)\s*-\s*(.+?)\s*-\s*XML TotalValues\.xml$", re.IGNORECASE)
    for p in sorted(pats):
        base = os.path.basename(p)
        m = rx.match(base)
        if not m: 
            continue
        fecha = m.group(1).strip()
        rival = m.group(2).strip()
        labels.append(f"Fecha {fecha} - {rival}")
    return labels

def rival_from_label(label: str) -> str:
    # "Fecha 8 - Union Ezpeleta" -> "Union Ezpeleta"
    parts = [p.strip() for p in label.split(" - ", 1)]
    return parts[1] if len(parts) == 2 else label

def infer_paths_for_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """
    A partir de 'Fecha N° - Rival' arma:
      XML_PATH:   data/minutos/Fecha N° - Rival - XML TotalValues.xml
      MATRIX_PATH:data/matrix/ Fecha N° - Rival - Matrix.xlsx (o .csv si existiera)
    """
    xml_path = os.path.join(DATA_MINUTOS, f"{label} - XML TotalValues.xml")
    mx_xlsx  = os.path.join(DATA_MATRIX,  f"{label} - Matrix.xlsx")
    mx_csv   = os.path.join(DATA_MATRIX,  f"{label} - Matrix.csv")
    matrix_path = mx_xlsx if os.path.isfile(mx_xlsx) else (mx_csv if os.path.isfile(mx_csv) else None)
    return (xml_path if os.path.isfile(xml_path) else None), matrix_path

# =========================
# PARSERS SEGÚN TU NOTEBOOK
# =========================
def parse_possession_from_equipo(xml_path: str) -> Tuple[float, float]:
    """Posesión desde XML (duración de instancias con code==tiempo posecion ferro/rival)."""
    if not xml_path or not os.path.isfile(xml_path): return 0.0, 0.0
    tree = ET.parse(xml_path)
    root = tree.getroot()
    t_ferro = t_rival = 0.0
    for inst in root.findall(".//instance"):
        code  = nlower(inst.findtext("code"))
        try:
            st = float(inst.findtext("start") or "0")
            en = float(inst.findtext("end") or "0")
        except:
            continue
        dur = max(0.0, en - st)
        if code == "tiempo posecion ferro":
            t_ferro += dur
        elif code == "tiempo posecion rival":
            t_rival += dur
    tot = t_ferro + t_rival
    if tot <= 0: return 0.0, 0.0
    return round(100*t_ferro/tot,1), round(100*t_rival/tot,1)

def load_matrix(path: str) -> Tuple[pd.DataFrame, str, Dict[str,str]]:
    if path.lower().endswith(".xlsx"):
        # requiere openpyxl si usás xlsx (agregalo a requirements.txt)
        df = pd.read_excel(path, header=0)
    elif path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path, header=0)
        except Exception:
            df = pd.read_csv(path, header=0, sep=";")
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

    mx = {
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
    return mx

def parse_instances_jugadores(xml_path: str):
    if not xml_path or not os.path.isfile(xml_path): return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out = []
    for inst in root.findall(".//instance"):
        code = ntext(inst.findtext("code"))
        labels_lc = [nlower(t.text) for t in inst.findall("./label/text")]
        xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
        ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
        x_end, y_end = (xs[-1], ys[-1]) if xs and ys else (None, None)
        out.append({"code": code, "labels_lc": labels_lc, "end_xy": (x_end, y_end)})
    return out

def is_pass_attempt(ev) -> bool:
    if re.match(r"^\s*pase\b", nlower(ev["code"])): return True
    return any(re.match(r"^\s*pase\b", l) for l in ev["labels_lc"])

def is_rival_code(code) -> bool:
    return nlower(code).startswith("categoria - equipo rival")

def xy_to_zone(x, y, max_x=19, max_y=34):
    if x is None or y is None: return None
    col = 1 if x <= 6 else (2 if x <= 13 else 3)
    row = 1 if y <= 11 else (2 if y <= 22 else 3)
    return (row-1)*3 + col  # 1..9

def passes_last_third_and_area(jug):
    last_m = last_r = area_m = area_r = 0
    for ev in jug:
        if not is_pass_attempt(ev):
            continue
        x, y = ev["end_xy"]
        z = xy_to_zone(x, y)
        if z is None:
            continue
        if is_rival_code(ev["code"]):
            if z in {1,4,7}: last_r += 1
            if z == 4: area_r += 1
        else:
            if z in {3,6,9}: last_m += 1
            if z == 6: area_m += 1
    return last_m, last_r, area_m, area_r

# =========================
# VISUAL (tu panel)
# =========================
def load_any_image(path):
    im = Image.open(path); im.load()
    if im.mode != "RGBA":
        im = im.convert("RGBA")
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
        if TRIM_LOGO_BORDERS:
            img = trim_margins(img)
    except Exception:
        return
    h, w = img.shape[0], img.shape[1]
    aspect = h / w if w else 1.0
    w_norm = width; h_norm = width * aspect
    ax.imshow(img, extent=[cx - w_norm/2, cx + w_norm/2,
                           cy - h_norm/2, cy + h_norm/2], zorder=6)

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

    # Banner inferior (firma)
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
        draw_row(y, lab, FERRO.get(lab,0), RIVAL.get(lab,0))
        y -= row_h

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
# UI — ESTADÍSTICAS PARTIDO
# =========================
menu = st.sidebar.radio(
    "Menú",
    ["📊 Estadísticas de partido", "⏱️ Minutos", "🎯 Tiros", "🗺️ Mapa 3x3", "🔗 Red de Pases", "⚡ Radar"],
    index=0
)

if menu == "📊 Estadísticas de partido":
    matches = list_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos con patrón: 'Fecha N° - Rival - XML TotalValues.xml'.")
        st.stop()

    sel = st.selectbox("Elegí partido", matches, index=0)
    rival = rival_from_label(sel)

    XML_PATH, MATRIX_PATH = infer_paths_for_label(sel)

    # 1) Posesión desde XML (TotalValues)
    pos_m, pos_r = parse_possession_from_equipo(XML_PATH) if XML_PATH else (0.0, 0.0)

    # 2) Matrix (métricas por equipo)
    if MATRIX_PATH and os.path.isfile(MATRIX_PATH):
        try:
            mx = compute_from_matrix(MATRIX_PATH)
        except Exception as e:
            st.error(f"No pude leer MATRIX '{os.path.basename(MATRIX_PATH)}': {e}")
            mx = {"FERRO": {}, "RIVAL": {}}
    else:
        mx = {"FERRO": {}, "RIVAL": {}}

    # 3) Jugadores (último tercio / al área) — usando el mismo XML TotalValues
    last_m = last_r = area_m = area_r = 0
    if XML_PATH and os.path.isfile(XML_PATH):
        jug = parse_instances_jugadores(XML_PATH)
        last_m, last_r, area_m, area_r = passes_last_third_and_area(jug)

    # Armar FERRO / RIVAL para el render (nombres exactos)
    FERRO = {
        "Posesión %":           float(pos_m),
        "Pases totales":        int(mx.get("FERRO", {}).get("Pases totales", 0)),
        "Pases OK %":           float(mx.get("FERRO", {}).get("Pases OK %", 0)),
        "Pases último tercio":  int(last_m),
        "Pases al área":        int(area_m),
        "Tiros":                int(mx.get("FERRO", {}).get("Tiros", 0)),
        "Tiros al arco":        int(mx.get("FERRO", {}).get("Tiros al arco", 0)),
        "Recuperaciones":       int(mx.get("FERRO", {}).get("Recuperaciones", 0)),
        "Duelos ganados":       int(mx.get("FERRO", {}).get("Duelos Ganados", 0)),
        "% Duelos ganados":     float(mx.get("FERRO", {}).get("% Duelos Ganados", 0)),
        "Corners":              int(mx.get("FERRO", {}).get("Corners", 0)),
        "Faltas":               int(mx.get("FERRO", {}).get("Faltas", 0)),
        "Goles":                int(mx.get("FERRO", {}).get("Goles", 0)),
        "Asistencias":          int(mx.get("FERRO", {}).get("Asistencias", 0)),
        "Pases clave":          int(mx.get("FERRO", {}).get("Pases Clave", 0)),
    }
    RIVAL = {
        "Posesión %":           float(pos_r),
        "Pases totales":        int(mx.get("RIVAL", {}).get("Pases totales", 0)),
        "Pases OK %":           float(mx.get("RIVAL", {}).get("Pases OK %", 0)),
        "Pases último tercio":  int(last_r),
        "Pases al área":        int(area_r),
        "Tiros":                int(mx.get("RIVAL", {}).get("Tiros", 0)),
        "Tiros al arco":        int(mx.get("RIVAL", {}).get("Tiros al arco", 0)),
        "Recuperaciones":       int(mx.get("RIVAL", {}).get("Recuperaciones", 0)),
        "Duelos ganados":       int(mx.get("RIVAL", {}).get("Duelos Ganados", 0)),
        "% Duelos ganados":     float(mx.get("RIVAL", {}).get("% Duelos Ganados", 0)),
        "Corners":              int(mx.get("RIVAL", {}).get("Corners", 0)),
        "Faltas":               int(mx.get("RIVAL", {}).get("Faltas", 0)),
        "Goles":                int(mx.get("RIVAL", {}).get("Goles", 0)),
        "Asistencias":          int(mx.get("RIVAL", {}).get("Asistencias", 0)),
        "Pases clave":          int(mx.get("RIVAL", {}).get("Pases Clave", 0)),
    }

    ferro_logo = badge_path_for("ferro")
    rival_logo = badge_path_for(rival)

    draw_key_stats_panel(
        home_name="FERRO",
        away_name=rival,
        FERRO=FERRO,
        RIVAL=RIVAL,
        ferro_logo_path=ferro_logo,
        rival_logo_path=rival_logo
    )

    with st.expander("Ver tabla base (debug)"):
        tbl = pd.DataFrame({"Métrica": ROW_ORDER,
                            "Ferro": [FERRO[k] for k in ROW_ORDER],
                            "Rival": [RIVAL[k]  for k in ROW_ORDER]})
        st.dataframe(tbl, use_container_width=True)

else:
    st.info("Las demás secciones se irán conectando con tus notebooks en los próximos pasos.")
