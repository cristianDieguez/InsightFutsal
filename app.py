# app.py — InsightFutsal: Estadísticas de Partido (panel KEY STATS)
# Lee XML desde data/minutos/, usa Matrix.xlsx si existe, render estilo notebook.

import os, re, glob
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from lxml import etree

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="InsightFutsal", page_icon="⚽", layout="wide")

DATA_DIR   = "data/minutos"
BADGE_DIR  = "images/equipos"
BANNER_DIR = "images/banner"

# Nombres EXACTOS que usás en el panel
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

# Códigos de posesión (XML instancias)
CODE_POSESION_FERRO = "tiempo posecion ferro"
CODE_POSESION_RIVAL = "tiempo posecion rival"

# =========================
# UTILIDADES
# =========================
def list_xml_files(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir): return []
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(".xml") and "totalvalues" in f.lower()]
    files.sort()
    return files

def pretty_match_name(fname: str) -> str:
    return re.sub(r"(?i)\s*-\s*xml\s*totalvalues\.xml$", "", fname)

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
    if not os.path.isdir(BANNER_DIR): return None
    for ext in ("png","jpg","jpeg","webp"):
        for p in glob.glob(os.path.join(BANNER_DIR, f"*.{ext}")):
            img = open_any(p)
            if img: return img
    return None

def badge_for(team: str) -> Optional[Image.Image]:
    if not os.path.isdir(BADGE_DIR): return None
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

def normtext(s: str) -> str:
    return (s or "").strip().lower()

# =========================
# PARSER XML (INSTANCIAS) → POSESIÓN
# =========================
@st.cache_data(show_spinner=False)
def possession_from_instances(xml_path: str) -> Tuple[float,float]:
    try:
        with open(xml_path, "rb") as fh:
            root = etree.fromstring(fh.read())
        inst = root.findall(".//ALL_INSTANCES/instance")
        if not inst: inst = root.findall(".//instance")
        own = away = 0.0
        for it in inst:
            code = normtext(it.findtext("code"))
            start = it.findtext("start"); end = it.findtext("end")
            dur = (float(end) - float(start)) if (start and end) else 0.0
            if CODE_POSESION_FERRO in code:
                own += dur
            elif CODE_POSESION_RIVAL in code:
                away += dur
        total = own + away
        if total <= 0: return 0.0, 0.0
        own_pct = round(own/total*100, 1)
        away_pct = round(100 - own_pct, 1)
        return own_pct, away_pct
    except Exception:
        return 0.0, 0.0

# =========================
# LOAD MATRIX.XLSX (PREFERIDO)
# =========================
def find_matrix_for_rival(rival: str) -> Optional[str]:
    # busca recursivamente algo tipo "*Union Ezpeleta*Matrix.xlsx"
    pat = f"*{re.sub(r'\\s+','*', rival)}*Matrix.xlsx"
    for p in glob.glob(f"**/{pat}", recursive=True):
        return p
    return None

def normalize_metric_name(s: str) -> str:
    return normtext(s).replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("%","%")

def coerce_number(x):
    try:
        if pd.isna(x): return 0
        if isinstance(x, str) and x.strip().endswith("%"):
            return float(x.strip().replace("%",""))
        return float(x)
    except Exception:
        return 0

def ferro_rival_from_matrix(xlsx_path: str) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Intenta leer una matriz con columnas tipo:
      - ['Métrica','FERRO','RIVAL']  o
      - ['metric','FERRO','RIVAL']   o
      - índice con métricas y columnas FERRO/RIVAL
    Devuelve dos dicts con los NOMBRES EXACTOS del panel.
    """
    FERRO = {k:0 for k in ROW_ORDER}
    RIVAL = {k:0 for k in ROW_ORDER}

    xl = pd.ExcelFile(xlsx_path)
    # busca la primera hoja que parezca tabla métrica
    sheet = xl.sheet_names[0]
    df = xl.parse(sheet)
    # normaliza nombres de columnas
    cols = [normtext(c) for c in df.columns]
    df.columns = cols

    # casos: métrica en primera col o como índice
    if ("ferro" in cols) and ("rival" in cols):
        # detectar col con nombre de métrica
        metric_col = None
        for c in df.columns:
            if c not in ("ferro","rival") and "metric" in c:
                metric_col = c; break
        if metric_col is None:
            metric_col = df.columns[0]  # asumimos la primera
        for _, r in df.iterrows():
            m  = str(r[metric_col]).strip()
            m0 = normalize_metric_name(m)
            # mapeo por similitud simple a nombre exacto
            for target in ROW_ORDER:
                if normalize_metric_name(target) == m0:
                    FERRO[target] = coerce_number(r["ferro"])
                    RIVAL[target]  = coerce_number(r["rival"])
                    break
    else:
        # probar que el índice sean métricas y haya columnas FERRO/RIVAL
        df = df.set_index(df.columns[0])
        df.columns = [normtext(c) for c in df.columns]
        if "ferro" in df.columns and "rival" in df.columns:
            for target in ROW_ORDER:
                key = normalize_metric_name(target)
                # buscar fila que matchee exacto
                for idx in df.index:
                    if normalize_metric_name(str(idx)) == key:
                        FERRO[target] = coerce_number(df.loc[idx, "ferro"])
                        RIVAL[target]  = coerce_number(df.loc[idx, "rival"])
                        break

    return FERRO, RIVAL

# =========================
# FALLBACK: TOTALVALUES (por si no hay matriz)
# =========================
def read_totalvalues_pairs(xml_path: str) -> Dict[str, float]:
    """
    Lee asXML TotalValues. Soporta varias formas:
      <Value label="Tiros" count="22"/>
      <Value><label>Tiros</label><count>22</count></Value>
    Devuelve dict label->valor
    """
    out = {}
    try:
        with open(xml_path, "rb") as fh:
            root = etree.fromstring(fh.read())
        for v in root.findall(".//Value"):
            lab = v.get("label") or v.findtext("label") or v.get("name") or v.findtext("Name") or ""
            cnt = v.get("count") or v.findtext("count") or v.get("value") or v.findtext("Value") or "0"
            lab = str(lab).strip()
            try:
                val = float(str(cnt).strip().replace(",",".").replace("%",""))
            except Exception:
                val = 0
            if lab:
                out[lab] = val
    except Exception:
        pass
    return out

def ferro_rival_from_totalvalues(xml_path: str, rival: str) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Intenta deducir FERRO/RIVAL a partir de un único TotalValues.
    (Si tenés uno por cada equipo, podés duplicar la lectura y fusionar).
    Aquí mapeamos por clave exacta.
    """
    stats = read_totalvalues_pairs(xml_path)
    FERRO = {k:0 for k in ROW_ORDER}
    RIVAL = {k:0 for k in ROW_ORDER}
    # si las claves ya vienen como FERRO/RIVAL en el archivo, podés adaptar acá.
    # por defecto dejamos todo en 0 salvo lo que sea claramente global.
    # (normalmente vas a tener la Matrix.xlsx y no vas a caer en este fallback)
    return FERRO, RIVAL

# =========================
# RENDER PANEL (como tu notebook)
# =========================
# Estilo
bg_green   = "#006633"
text_w     = "#FFFFFF"
bar_white  = "#FFFFFF"
bar_rival  = "#E6EEF2"
bar_rail   = "#0F5E29"
star_c     = "#FFD54A"
loser_alpha = 0.35
orange_win = "#FF8F00"

import matplotlib as mpl
mpl.rcParams.update({
    "savefig.facecolor": bg_green,
    "figure.facecolor":  bg_green,
    "axes.facecolor":    bg_green,
    "text.color":        text_w,
})

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

def draw_key_stats_panel(home_name: str, away_name: str, FERRO: Dict[str,float], RIVAL: Dict[str,float],
                         ferro_logo_path: Optional[str], rival_logo_path: Optional[str],
                         footer_left: Optional[str], footer_right: Optional[str]):

    plt.close("all")
    fig_h = 0.66*len(ROW_ORDER) + 4.6
    fig = plt.figure(figsize=(10.8, fig_h))
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(Rectangle((0,0), 1, 1, facecolor=bg_green, edgecolor="none", zorder=-10))

    # banner sup.
    BANNER_Y0 = 1.0 - (BANNER_H + 0.02)
    ax.add_patch(Rectangle((0, BANNER_Y0), 1, BANNER_H, facecolor="white", edgecolor="none", zorder=5))
    if ferro_logo_path: draw_logo(ax, ferro_logo_path, 0.09, BANNER_Y0 + BANNER_H*0.52, LOGO_W)
    if rival_logo_path: draw_logo(ax, rival_logo_path, 0.91, BANNER_Y0 + BANNER_H*0.52, LOGO_W)
    ax.text(0.5, BANNER_Y0 + BANNER_H*0.63, f"{home_name.upper()} vs {away_name.upper()}",
            ha="center", va="center", fontsize=TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, BANNER_Y0 + BANNER_H*0.29, "KEY STATS",
            ha="center", va="center", fontsize=SUB_FS, weight="bold", color=bg_green, zorder=7)

    # banner inferior
    FOOTER_Y0 = 0.02
    ax.add_patch(Rectangle((0, FOOTER_Y0), 1, FOOTER_H, facecolor="white", edgecolor="none", zorder=5))
    if footer_left:  draw_logo(ax, footer_left,  0.09, FOOTER_Y0 + FOOTER_H*0.52, FOOTER_LOGO_W)
    if footer_right: draw_logo(ax, footer_right, 0.91, FOOTER_Y0 + FOOTER_H*0.52, FOOTER_LOGO_W)
    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.63, "Trabajo Fin de Máster",
            ha="center", va="center", fontsize=FOOTER_TITLE_FS, weight="bold", color=bg_green, zorder=7)
    ax.text(0.5, FOOTER_Y0 + FOOTER_H*0.28, "Cristian Dieguez",
            ha="center", va="center", fontsize=FOOTER_SUB_FS,   weight="bold", color=bg_green, zorder=7)

    # cuerpo
    EXTRA_GAP_BELOW_BANNER = 0.075
    top_y    = BANNER_Y0 - EXTRA_GAP_BELOW_BANNER
    bottom_y = FOOTER_Y0 + FOOTER_H + 0.012
    available_h = max(0.01, top_y - bottom_y)
    row_h = available_h / len(ROW_ORDER)

    mid_x, bar_w = 0.5, 0.33
    left_star_x, right_star_x = 0.055, 0.945

    def draw_row(y, label, lv, rv):
        label_y = y + row_h*(LABEL_Y_SHIFT_HIGH if RAISE_LABELS else LABEL_Y_SHIFT_LOW)
        ax.text(0.5, label_y, label, ha="center", va="center", fontsize=12.5, weight="bold", color=text_w)

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

        ax.add_patch(Rectangle((mid_x - 0.02 - lw, lane_y), lw, lane_h,
                               facecolor=ferro_col, edgecolor=ferro_col, alpha=ferro_alpha))
        ax.add_patch(Rectangle((mid_x + 0.02, lane_y), rw, lane_h,
                               facecolor=rival_col, edgecolor=rival_col, alpha=rival_alpha))

        ltxt = f"{lv:.1f}%" if label in PERCENT_ROWS else f"{int(lv)}"
        rtxt = f"{rv:.1f}%" if label in PERCENT_ROWS else f"{int(rv)}"
        ax.text(mid_x - bar_w - 0.030, lane_y + lane_h*0.45, ltxt, ha="right", va="center",
                fontsize=13, weight="bold", color=text_w)
        ax.text(mid_x + bar_w + 0.030, lane_y + lane_h*0.45, rtxt, ha="left", va="center",
                fontsize=13, weight="bold", color=text_w)

        if winner == "FERRO":
            ax.text(left_star_x,  label_y, "★", ha="left",  va="center", fontsize=14, color=star_c, weight="bold", clip_on=False)
        elif winner == "RIVAL":
            ax.text(right_star_x, label_y, "★", ha="right", va="center", fontsize=14, color=star_c, weight="bold", clip_on=False)

    y = top_y
    for lab in ROW_ORDER:
        draw_row(y, lab, FERRO.get(lab,0), RIVAL.get(lab,0))
        y -= row_h

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================
# UI
# =========================
st.sidebar.header("Opciones")
menu = st.sidebar.radio(
    "Menú",
    ["📊 Estadísticas de Partido", "⏱️ Minutos", "🎯 Tiros", "🗺️ Mapa 3x3", "🔗 Red de Pases", "⚡ Radar"],
    index=0
)

banner = first_banner()
if banner is not None:
    st.image(banner, use_container_width=True)

st.title("InsightFutsal")

if menu == "📊 Estadísticas de Partido":
    files = list_xml_files(DATA_DIR)
    if not files:
        st.error(f"No hay XML con 'TotalValues' en {DATA_DIR}")
        st.stop()

    pretty = [pretty_match_name(f) for f in files]
    sel_pretty = st.selectbox("Elegí partido", pretty, index=0)
    fname = files[pretty.index(sel_pretty)]

    rival = extract_rival(fname)
    xml_path = os.path.join(DATA_DIR, fname)

    # 1) Posesión desde instancias
    pos_own, pos_away = possession_from_instances(xml_path)

    # 2) Matriz (preferido)
    FERRO = {k:0 for k in ROW_ORDER}
    RIVAL = {k:0 for k in ROW_ORDER}
    FERRO["Posesión %"] = pos_own
    RIVAL["Posesión %"] = pos_away

    mx_path = find_matrix_for_rival(rival)
    if mx_path:
        f2, r2 = ferro_rival_from_matrix(mx_path)
        # respetamos nombres exactos del panel
        for k in ROW_ORDER:
            if k in f2: FERRO[k] = f2[k] if k in PERCENT_ROWS else int(round(f2[k]))
            if k in r2: RIVAL[k]  = r2[k]  if k in PERCENT_ROWS else int(round(r2[k]))
    else:
        # 3) Fallback a TotalValues si no hay matriz (deja 0 si no encuentra)
        f2, r2 = ferro_rival_from_totalvalues(xml_path, rival)
        for k in ROW_ORDER:
            if k in f2: FERRO[k] = f2[k] if k in PERCENT_ROWS else int(round(f2[k]))
            if k in r2: RIVAL[k]  = r2[k]  if k in PERCENT_ROWS else int(round(r2[k]))

    # Logos
    ferro_logo = badge_for("Ferro")
    ferro_logo_path = None
    if ferro_logo:
        # no tengo path directo del PIL; busco primero que matchee
        for ext in ("png","jpg","jpeg","webp"):
            p = os.path.join(BADGE_DIR, f"ferro.{ext}")
            if os.path.isfile(p): ferro_logo_path = p; break

    rival_logo_path = None
    for ext in ("png","jpg","jpeg","webp"):
        p = os.path.join(BADGE_DIR, f"{rival.lower().replace(' ','_')}.{ext}")
        if os.path.isfile(p): rival_logo_path = p; break
        p = os.path.join(BADGE_DIR, f"{rival.lower().replace(' ','')}.{ext}")
        if os.path.isfile(p): rival_logo_path = p; break
        p = os.path.join(BADGE_DIR, f"{rival}.{ext}")
        if os.path.isfile(p): rival_logo_path = p; break

    footer_left = footer_right = None
    # si tenés logos específicos en images/banner ponelos con el nombre que quieras; acá no forzamos

    draw_key_stats_panel(
        home_name="FERRO",
        away_name=rival.upper(),
        FERRO=FERRO,
        RIVAL=RIVAL,
        ferro_logo_path=ferro_logo_path,
        rival_logo_path=rival_logo_path,
        footer_left=None,
        footer_right=None
    )

    with st.expander("Ver tabla base (debug)"):
        tbl = pd.DataFrame({"Métrica": ROW_ORDER,
                            "Ferro": [FERRO[k] for k in ROW_ORDER],
                            "Rival": [RIVAL[k]  for k in ROW_ORDER]})
        st.dataframe(tbl, use_container_width=True)

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
