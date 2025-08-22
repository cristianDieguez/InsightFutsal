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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle as MplCircle


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
# TIMELINE — helpers y parsers
# =========================
from matplotlib.patches import Circle

BALL_ICON_PATH    = os.path.join(BANNER_DIR, "pelota.png")
FOOTER_LEFT_LOGO  = os.path.join(BANNER_DIR, "SportData.png")
FOOTER_RIGHT_LOGO = os.path.join(BANNER_DIR, "Sevilla.png")
INCLUDE_GOALS_IN_SHOT_CIRCLES = True

def is_shot(ev):
    s = nlower(ev["code"])
    if re.search(r"\btiro\b|\bremate\b", s): return True
    return any(re.search(r"\btiro\b|\bremate\b", l) for l in ev["labels_lc"])

def is_goal(ev):
    s = nlower(ev["code"])
    if re.match(r"^gol\b", s): return True
    return any(re.match(r"^gol\b", l) for l in ev["labels_lc"])

ON_TARGET_PAT = re.compile(
    r"\b(al\s*arco|a\s*puerta|a\s*porter[ií]a|on\s*target|atajad[oa]|saved\s*shot)\b", re.IGNORECASE
)
def is_shot_on_target(ev):
    s = nlower(ev["code"])
    if ON_TARGET_PAT.search(s): return True
    return any(ON_TARGET_PAT.search(l or "") for l in ev["labels_lc"])

def minute_bucket(sec):
    m = int(sec // 60)
    return max(0, min(39, m))  # 0..39

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

        # pases + último tercio
        if is_pass_attempt(ev):
            if is_rival_code(ev["code"]): tl["passes_R"][m] += 1
            else:                          tl["passes_M"][m] += 1
            z = xy_to_zone(*(ev["end_xy"] or (None, None)))
            if z is not None:
                if is_rival_code(ev["code"]):
                    if z in {1,4,7}: tl["last_R"][m] += 1
                else:
                    if z in {3,6,9}: tl["last_M"][m] += 1

        # tiros / on target / goles
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
yellow_on  = "#FFD54F"
rival_g    = "#B9C4C9"
white      = "#FFFFFF"
rail       = "#0F5E29"

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

def draw_count_circle(ax, x, y, count, base_r=0.006, face=None, edge=white, lw=1.0, z=8):
    if count <= 0: return
    from math import sqrt
    r = base_r * sqrt(count ** 0.5)
    circ = Circle((x, y), radius=r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(circ)

def draw_timeline_panel(rival_name: str, tl: dict,
                        ferro_logo_path: Optional[str], rival_logo_path: Optional[str]):
    plt.close("all")
    fig_h = 11.8
    fig = plt.figure(figsize=(10.8, fig_h))
    ax  = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
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

    x_center_gap_L = 0.47
    x_center_gap_R = 0.53
    x_bar_M_max = 0.22
    x_bar_R_max = 0.78
    x_shot_M  = 0.05
    x_last_M  = 0.16
    x_shot_R  = 0.95
    x_last_R  = 0.84
    x_goal_M  = x_shot_M - 0.025
    x_goal_R  = x_shot_R + 0.025

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
    def y_of_index(i):
        return panel_y1 - panel_h * (i + 0.5) / M

    ax.add_line(plt.Line2D([x_center_gap_L, x_center_gap_L], [panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
    ax.add_line(plt.Line2D([x_center_gap_R, x_center_gap_R], [panel_y0, panel_y1], color=white, alpha=0.28, lw=1.2))
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

        # FERRO izquierda
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

        # RIVAL derecha
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

@st.cache_data(show_spinner=False)
def discover_matches() -> List[Dict]:
    if not os.path.isdir(DATA_MINUTOS):
        return []

    # Solo buscamos XML NacSport y, si no existe para ese label, TotalValues
    pats = []
    pats += glob.glob(os.path.join(DATA_MINUTOS, "* - XML NacSport.xml"))
    pats += glob.glob(os.path.join(DATA_MINUTOS, "* - XML TotalValues.xml"))

    files = sorted(set(pats))

    # agrupar por label: “Fecha X - Rival”
    buckets: Dict[str, List[str]] = {}
    for p in files:
        base = os.path.basename(p)
        label = re.sub(r"(?i)\s*-\s*xml\s*(nacsport|totalvalues)\.xml$", "", base).strip()
        buckets.setdefault(label, []).append(p)

    def pref_key(path: str) -> int:
        b = os.path.basename(path).lower()
        # prioridad: NacSport (0) luego TotalValues (1)
        if "xml nacsport" in b:   return 0
        if "totalvalues" in b:    return 1
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
            "xml_players": pick,   # NacSport si hay; si no, TotalValues
            "rival": rival,
            "matrix_path": matrix_path,
            "xml_equipo": None,    # no lo usamos en estas vistas
        })

    # ordenar por número de fecha si aparece, si no alfabético
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
# HEATMAPS — helpers
# =========================

# Keywords de pases (mismo criterio que tu notebook)
KEYWORDS_PASES = [
    "pase corto frontal", "pase corto lateral",
    "pase largo frontal", "pase largo lateral",
    "pase progresivo frontal", "pase progresivo lateral",
    "pase progresivo frontal cpie", "pase progresivo lateral cpie",
    "salida de arco progresivo cmano",
    "pase corto frontal cpie", "pase corto lateral cpie",
    "salida de arco corto cmano"
]
KEYWORDS_PASES = [k.lower() for k in KEYWORDS_PASES]

def cargar_datos_nacsport(xml_path: str) -> pd.DataFrame:
    """Lee el XML (TotalValues) y devuelve DF con columnas:
       jugador (code), pos_x_list, pos_y_list, labels(list)
    """
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
    is_shot = any("tiro" in l for l in lbls) or any("remate" in l for l in lbls)
    is_pass = any(any(k in l for k in KEYWORDS_PASES) for l in lbls)
    return is_shot or is_pass

def filtrar_coords_por_evento(row: pd.Series) -> Tuple[List[float], List[float]]:
    """Si es pase (keywords) o tiro -> solo primer punto; si no, todos."""
    if _es_evento_pase_o_tiro(row.get("labels", [])):
        return row["pos_x_list"][:1], row["pos_y_list"][:1]
    return row["pos_x_list"], row["pos_y_list"]

def parse_player_and_role(code: str) -> Tuple[str, Optional[str]]:
    """'Bruno (Ala I)' -> ('Bruno','Ala I');  'Bruno' -> ('Bruno',None)"""
    m = re.match(r"^(.*)\((.*)\)\s*$", code.strip())
    if m:
        name = ntext(m.group(1)).strip()
        role = ntext(m.group(2)).strip()
        return name, role
    return code.strip(), None

def explode_coords_for_heatmap(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Devuelve DF con columnas: player_name, role, pos_x, pos_y"""
    if df_raw.empty:
        return pd.DataFrame(columns=["player_name","role","pos_x","pos_y"])

    tmp = df_raw.copy()
    tmp["pos_x_list"], tmp["pos_y_list"] = zip(*tmp.apply(filtrar_coords_por_evento, axis=1))
    tmp = tmp.explode(["pos_x_list","pos_y_list"], ignore_index=True)
    tmp = tmp.rename(columns={"pos_x_list":"pos_x","pos_y_list":"pos_y"})
    tmp = tmp.dropna(subset=["pos_x","pos_y"])

    names_roles = tmp["jugador"].apply(parse_player_and_role)
    tmp["player_name"] = names_roles.apply(lambda t: t[0])
    tmp["role"]        = names_roles.apply(lambda t: t[1])

    # normalización de strings
    tmp["player_name"] = tmp["player_name"].map(ntext)
    tmp["role"]        = tmp["role"].map(lambda x: ntext(x) if x else None)

    return tmp[["player_name","role","pos_x","pos_y"]]

# Cancha futsal horizontal (35 x 20)
def draw_futsal_pitch_horizontal(ax):
    ax.set_facecolor("white")
    ancho = 35  # largo
    alto  = 20  # ancho

    ax.plot([0, ancho],[0,0], color="black")
    ax.plot([0, ancho],[alto,alto], color="black")
    ax.plot([0,0],[0,alto], color="black")
    ax.plot([ancho,ancho],[0,alto], color="black")

    ax.plot([ancho/2, ancho/2],[0,alto], color="black")

    area_izq = Arc((0, alto/2), width=8, height=12, angle=0, theta1=270, theta2=90, color="black")
    ax.add_patch(area_izq)
    ax.plot([0,1.0],[8.5,8.5], color="black", linewidth=2)
    ax.plot([0,1.0],[11.5,11.5], color="black", linewidth=2)
    ax.plot([1.0,1.0],[8.5,11.5], color="black", linewidth=2)

    area_der = Arc((ancho, alto/2), width=8, height=12, angle=0, theta1=90, theta2=270, color="black")
    ax.add_patch(area_der)
    ax.plot([ancho, ancho-1.0],[8.5,8.5], color="black", linewidth=2)
    ax.plot([ancho, ancho-1.0],[11.5,11.5], color="black", linewidth=2)
    ax.plot([ancho-1.0, ancho-1.0],[8.5,11.5], color="black", linewidth=2)

    centre_circle = MplCircle((ancho/2, alto/2), 4, color="black", fill=False)
    centre_spot   = MplCircle((ancho/2, alto/2), 0.2, color="black")
    ax.add_patch(centre_circle); ax.add_patch(centre_spot)

    ax.annotate('', xy=(ancho-2, 2), xytext=(ancho-6, 2),
                arrowprops=dict(facecolor='blue', arrowstyle='->', lw=3))
    ax.text(ancho-2, 3, 'Sentido de Ataque', color='blue', fontsize=10, ha='right')

    ax.set_xlim(0, ancho); ax.set_ylim(0, alto); ax.axis('off')

def rotate_coords_for_attack_right(df: pd.DataFrame, role: Optional[str]=None) -> pd.DataFrame:
    """Rota coords del XML (20x35) a cancha horizontal (35x20).
       Y además invierte Y para Ala I / Ala D (ajuste táctico).
    """
    if df.empty: return df.copy()
    ancho_original = 20
    alto_original  = 35
    out = df.copy()
    out["x_rot"] = alto_original - out["pos_y"]
    out["y_rot"] = out["pos_x"]
    if role and role in {"Ala I","Ala D"}:
        out["y_rot"] = ancho_original - out["y_rot"]
    return out

def fig_heatmap(df_xy: pd.DataFrame, titulo: str) -> plt.Figure:
    """Genera la figura de heatmap pastel sobre cancha."""
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_futsal_pitch_horizontal(ax)
    if df_xy.empty:
        ax.text(0.5, 0.5, "Sin datos para el filtro", ha="center", va="center", transform=ax.transAxes)
        return fig

    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_heatmap", ["#a8dadc", "#bde0c6", "#ffe5a1", "#f6a192"]
    )
    # límites del clip en sistema rotado: x∈[0,35], y∈[0,20]
    sns.kdeplot(
        x=df_xy["x_rot"], y=df_xy["y_rot"],
        fill=True, cmap=pastel_cmap, bw_adjust=0.4,
        levels=50, alpha=0.75, ax=ax,
        clip=((0, 35), (0, 20))
    )
    ax.set_title(titulo, fontsize=14)
    return fig

# =========================
# MINUTOS — helpers (desde tu notebook, adaptado)
# =========================
_NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")

def _split_name_role(code: str):
    m = _NAME_ROLE_RE.match(str(code)) if code else None
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)

def _merge_intervals(intervals):
    ints = [(float(s), float(e)) for s, e in intervals if s is not None and e is not None and e > s]
    if not ints: return []
    ints.sort()
    merged = [list(ints[0])]
    for s, e in ints[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def cargar_equipo_presencias(xml_path: str) -> pd.DataFrame:
    """Lee XML del equipo. Devuelve: code, nombre, rol, start_s, end_s, dur_s."""
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["code","nombre","rol","start_s","end_s","dur_s"])
    root = ET.parse(xml_path).getroot()
    rows = []
    for inst in root.findall(".//instance"):
        code = inst.findtext("code")
        nombre, rol = _split_name_role(code)
        if nombre is None:
            continue
        st = inst.findtext("start"); en = inst.findtext("end")
        try:
            s = float(st) if st is not None else None
            e = float(en) if en is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s:
            continue
        rows.append({"code": code, "nombre": nombre, "rol": rol,
                     "start_s": s, "end_s": e, "dur_s": e - s})
    return pd.DataFrame(rows)

def _format_mmss(seconds: float | int) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "00:00"
    s = int(round(float(seconds)))
    mm, ss = divmod(s, 60)
    return f"{mm:02d}:{ss:02d}"

def minutos_por_presencia(df_pres: pd.DataFrame):
    """Une solapes y suma minutos exactos por (nombre, rol) y por jugador."""
    if df_pres.empty:
        return (pd.DataFrame(columns=["nombre","rol","segundos","mmss","minutos","n_tramos"]),
                pd.DataFrame(columns=["nombre","segundos","mmss","minutos","n_tramos"]))
    out = []
    for (nombre, rol), g in df_pres.groupby(["nombre","rol"], dropna=False):
        intervals = list(zip(g["start_s"], g["end_s"]))
        merged = _merge_intervals(intervals)
        secs = sum(e - s for s, e in merged)
        out.append({
            "nombre": nombre,
            "rol": rol,
            "segundos": int(round(secs)),
            "mmss": _format_mmss(secs),
            "minutos": secs/60.0,
            "n_tramos": len(merged)
        })
    df_por_rol = (pd.DataFrame(out)
                    .sort_values(["segundos","nombre"], ascending=[False, True])
                    .reset_index(drop=True))
    df_por_rol["minutos"] = df_por_rol["minutos"].round(2)

    df_por_jugador = (df_por_rol
                      .groupby("nombre", as_index=False)
                      .agg(segundos=("segundos","sum"),
                           minutos=("minutos","sum"),
                           n_tramos=("n_tramos","sum"))
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
        txt_x = x * 0.98
        if x < 0.9:
            txt_x = min(x + 0.1, xlim * 0.98)
            ha = "left"; color = "black"; fw="bold"
        else:
            ha = "right"; color = "white"; fw="bold"
        ax.text(txt_x, y, mmss, va="center", ha=ha, fontsize=9, color=color, fontweight=fw)

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

# ========= Minutos desde XML TotalValues (sin XML de Equipo) =========
# ========= Filtros de códigos =========
_EXCLUDE_PREFIXES = tuple([  # normalizados a lower y sin tildes
    "categoria - equipo rival",
    "tiempo posecion ferro", "tiempo posesion ferro",
    "tiempo posecion rival", "tiempo posesion rival",
    "tiempo no jugado",
])
_NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")

def _strip_accents(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", str(s))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def is_player_code(code: str) -> bool:
    """True si code es 'Nombre (Rol)' válido y NO está en la lista de excluidos."""
    if not code: return False
    code_norm = _strip_accents(code).lower().strip()
    if any(code_norm.startswith(pref) for pref in _EXCLUDE_PREFIXES):
        return False
    return _NAME_ROLE_RE.match(code) is not None

def _split_name_role_if_player(code: str):
    """
    Devuelve (nombre, rol) si code es 'Nombre (Rol)' válido y NO pertenece a excluidos.
    Si no matchea, devuelve (None, None).
    """
    if not code:
        return (None, None)
    code_norm = _strip_accents(code).lower().strip()
    if any(code_norm.startswith(pref) for pref in _EXCLUDE_PREFIXES):
        return (None, None)
    m = _NAME_ROLE_RE.match(code)
    if not m:
        return (None, None)
    return (m.group(1).strip(), m.group(2).strip())

def cargar_minutos_desde_xml_totalvalues(xml_path: str) -> pd.DataFrame:
    """
    Lee el XML (NacSport o TotalValues) y devuelve presencias:
    code, nombre, rol, start_s, end_s, dur_s — sólo 'Jugador (Rol)' (excluye rivales/tiempos).
    """
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["code","nombre","rol","start_s","end_s","dur_s"])

    root = ET.parse(xml_path).getroot()
    rows = []
    for inst in root.findall(".//instance"):
        code = inst.findtext("code") or ""
        if not is_player_code(code):
            continue
        m = _NAME_ROLE_RE.match(code); nombre, rol = m.group(1).strip(), m.group(2).strip()
        st = inst.findtext("start"); en = inst.findtext("end")
        try:
            s = float(st) if st is not None else None
            e = float(en) if en is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s:
            continue
        rows.append({"code": code, "nombre": nombre, "rol": rol, "start_s": s, "end_s": e, "dur_s": e - s})
    return pd.DataFrame(rows)

# =========================
# RED DE PASES — helpers
# =========================
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

# cancha futsal horizontal 35x20
def draw_futsal_pitch_horizontal(ax):
    ax.set_facecolor("white")
    ancho, alto = 35, 20
    ax.plot([0, ancho], [0, 0], color="black")
    ax.plot([0, ancho], [alto, alto], color="black")
    ax.plot([0, 0], [0, alto], color="black")
    ax.plot([ancho, ancho], [0, alto], color="black")
    ax.plot([ancho / 2, ancho / 2], [0, alto], color="black")
    # áreas y marcas
    ax.add_patch(Arc((0, alto/2), 8, 12, angle=0, theta1=270, theta2=90, color="black"))
    ax.plot([0, 1], [8.5, 8.5], color="black", linewidth=2)
    ax.plot([0, 1], [11.5, 11.5], color="black", linewidth=2)
    ax.plot([1, 1], [8.5, 11.5], color="black", linewidth=2)
    ax.add_patch(Arc((ancho, alto/2), 8, 12, angle=0, theta1=90, theta2=270, color="black"))
    ax.plot([ancho, ancho-1], [8.5, 8.5], color="black", linewidth=2)
    ax.plot([ancho, ancho-1], [11.5, 11.5], color="black", linewidth=2)
    ax.plot([ancho-1, ancho-1], [8.5, 11.5], color="black", linewidth=2)
    ax.add_patch(Circle((ancho/2, alto/2), 4, color="black", fill=False))
    ax.add_patch(Circle((ancho/2, alto/2), 0.2, color="black"))
    ax.set_xlim(0, ancho); ax.set_ylim(0, alto); ax.axis("off")

def cargar_datos_nacsport(xml_path: str) -> pd.DataFrame:
    """Carga code, labels y listas de coordenadas desde XML TotalValues."""
    root = ET.parse(xml_path).getroot()
    data = []
    for inst in root.findall(".//instance"):
        jugador = inst.findtext("code") or ""
        labels = [lbl.findtext("text") for lbl in inst.findall("label")]
        pos_x = [float(px.text) for px in inst.findall("pos_x") if px.text is not None]
        pos_y = [float(py.text) for py in inst.findall("pos_y") if py.text is not None]
        data.append({"jugador": jugador, "labels": labels, "pos_x_list": pos_x, "pos_y_list": pos_y})
    return pd.DataFrame(data)

# keywords de pase (ajústalas si cambian tus etiquetas)
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

def red_de_pases_por_rol(df: pd.DataFrame):
    from collections import defaultdict
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_futsal_pitch_horizontal(ax)

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

    coords_origen = defaultdict(list)
    totales_hechos_por_rol = defaultdict(int)
    conteo_roles_total = defaultdict(int)

    for _, row in df.iterrows():
        code = row.get("jugador") or ""
        if not is_player_code(code):
            continue  # sólo “Nombre (Rol)”
        labels_lower = [(_strip_accents(lbl).lower() if lbl else "") for lbl in row.get("labels", [])]
        if not any(k in lbl for lbl in labels_lower for k in PASS_KEYWORDS):
            continue

        # rol de origen
        rol_origen = _NAME_ROLE_RE.match(code).group(2).strip()

        # rol destino: primera etiqueta con (Rol) distinta a origen
        rol_destino = None
        for lbl in row.get("labels", []):
            if lbl and "(" in lbl and ")" in lbl:
                pos = lbl.split("(")[1].replace(")", "").strip()
                if pos and pos != rol_origen:
                    rol_destino = pos
                    break

        # coordenada de inicio del pase (rotación a 35x20)
        px, py = row.get("pos_x_list") or [], row.get("pos_y_list") or []
        if px and py:
            x0 = 35 - (py[0] * (35.0 / 40.0))  # y original (0..40) -> x cancha 35
            y0 = px[0]                         # x original (0..20) -> y cancha 20
            coords_origen[rol_origen].append((x0, y0))

        totales_hechos_por_rol[rol_origen] += 1
        if rol_destino and rol_destino != rol_origen:
            conteo_roles_total[tuple(sorted([rol_origen, rol_destino]))] += 1

    # promedios por rol
    rol_coords = {}
    for rol, coords in coords_origen.items():
        arr = np.array(coords)
        rol_coords[rol] = (arr[:,0].mean(), arr[:,1].mean())

    # ajustes heurísticos
    if "Arq" in rol_coords:
        rol_coords["Arq"] = (3, 10)
    if "Ala I" in rol_coords and "Ala D" in rol_coords:
        yI, yD = rol_coords["Ala I"][1], rol_coords["Ala D"][1]
        if yI < yD:
            rol_coords["Ala I"], rol_coords["Ala D"] = rol_coords["Ala D"], rol_coords["Ala I"]
            # swap en conteos y totales
            from collections import defaultdict as dd
            new_counts = dd(int)
            for (a,b), v in conteo_roles_total.items():
                aa = "Ala D" if a=="Ala I" else ("Ala I" if a=="Ala D" else a)
                bb = "Ala D" if b=="Ala I" else ("Ala I" if b=="Ala D" else b)
                new_counts[tuple(sorted([aa,bb]))] += v
            conteo_roles_total = new_counts
            new_tot = dd(int)
            for r,t in totales_hechos_por_rol.items():
                if r=="Ala I": new_tot["Ala D"] = t
                elif r=="Ala D": new_tot["Ala I"] = t
                else: new_tot[r] = t
            totales_hechos_por_rol = new_tot

    # dibujar edges
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
# PÉRDIDAS & RECUPERACIONES — HELPERS
# =========================
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle, Arc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Parámetros cancha
PR_ANCHO, PR_ALTO = 35, 20
PR_N_COLS, PR_N_ROWS = 3, 3
PR_ROLES_GK = {"arquero","portero","guardameta","gk","portiere"}

# Regex nombre (rol)
_PR_NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")

# Filtros de códigos (normalizar sin tildes + lower antes de comparar)
_PR_EXCLUDE_PREFIXES = (
    "categoria - equipo rival",
    "tiempo posecion ferro","tiempo posesion ferro",
    "tiempo posecion rival","tiempo posesion rival",
    "tiempo no jugado",
)

def pr_strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def pr_is_player_code(code: str) -> bool:
    if not code: return False
    code_norm = pr_strip_accents(code).lower().strip()
    if any(code_norm.startswith(pref) for pref in _PR_EXCLUDE_PREFIXES):
        return False
    return _PR_NAME_ROLE_RE.match(code) is not None

def pr_split_name_role(code: str):
    m = _PR_NAME_ROLE_RE.match(str(code)) if code else None
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)

def pr_cargar_datos(xml_path: str) -> pd.DataFrame:
    """Carga XML NacSport o TotalValues (mismo formato de tags)"""
    root = ET.parse(xml_path).getroot()
    data = []
    for inst in root.findall(".//instance"):
        jugador = inst.findtext("code") or ""
        # Nos quedamos solo con 'Nombre (Rol)'
        if not pr_is_player_code(jugador):
            continue
        pos_x = [float(px.text) for px in inst.findall("pos_x") if (px.text or "").strip()!=""]
        pos_y = [float(py.text) for py in inst.findall("pos_y") if (py.text or "").strip()!=""]
        labels = [ (lbl.findtext("text") or "").strip() for lbl in inst.findall("label") ]
        st = inst.findtext("start"); en = inst.findtext("end")
        try:
            start = float(st) if st is not None else None
            end   = float(en) if en is not None else None
        except Exception:
            start, end = None, None
        data.append({
            "jugador": jugador,
            "labels": [l for l in labels if l],
            "pos_x_list": pos_x, "pos_y_list": pos_y,
            "start": start, "end": end
        })
    return pd.DataFrame(data)

def pr_cargar_presencias_equipo(xml_path: str) -> pd.DataFrame:
    """Si existe XML de equipo (presencias por jugador/rol). Si no existe, devolvemos DF vacío."""
    if not xml_path or not os.path.isfile(xml_path):
        return pd.DataFrame(columns=["nombre","rol","start_s","end_s"])
    root = ET.parse(xml_path).getroot()
    rows = []
    for inst in root.findall(".//instance"):
        code = inst.findtext("code") or ""
        nombre, rol = pr_split_name_role(code)
        if not nombre: 
            continue
        st = inst.findtext("start"); en = inst.findtext("end")
        try:
            s = float(st) if st is not None else None
            e = float(en) if en is not None else None
        except Exception:
            s, e = None, None
        if s is None or e is None or e <= s:
            continue
        rows.append({"nombre": nombre, "rol": rol, "start_s": s, "end_s": e})
    return pd.DataFrame(rows)

# Normalizaciones
def pr_labels_low(labels): return [pr_strip_accents(l).lower() for l in (labels or [])]
def pr_transform_xy(px, py):  # rotación/escala al mismo criterio del resto de visus
    return PR_ANCHO - (py * (PR_ANCHO / 40.0)), px

def pr_rol_de(code):
    if code and "(" in code and ")" in code:
        return code.split("(")[1].split(")")[0].strip()
    return None

def pr_ajustar_ala(y, code):
    if pr_rol_de(code) in ("Ala I","Ala D"):
        return PR_ALTO - y
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

# Palabras clave
_PR_KW_TOTAL   = [k.lower() for k in ["pase","recuperacion","recuperación","perdida","pérdida","pierde","conseguido","faltas","centro","lateral","despeje","despeje rival","gira","aguanta","cpie","cmano"]]
_PR_KW_PERDIDA = [k.lower() for k in ["perdida","pérdida","pierde","despeje"]]
_PR_KW_RECU    = [k.lower() for k in ["recuperacion","recuperación","1v1 ganado","despeje rival","cpie","cmano"]]
_PR_KW_TIRO_NOGOL  = ["desviado","errado","pifia","ataj","bloque","poste","travesa"]
_PR_KW_GOL_EXCLUDE = ["gol"]

def pr_contiene(labels_low, kws):
    return any(k in l for l in labels_low for k in kws)

def pr_zone_for(evento, code, pxs, pys):
    # Para PERDIDA usamos idx destino (1) si existe; para RECUPERACION, idx origen (0)
    if evento == "PERDIDA":
        idx = 1 if (pxs and len(pxs)>1) else 0
    else:
        idx = 0
    p = pr_get_point(pxs, pys, idx)
    if p is None: return None
    x, y = pr_transform_xy(*p)
    y = pr_ajustar_ala(y, code)
    return pr_zona_from_xy(x, y)

def pr_merge_minutes(df_pres: pd.DataFrame) -> dict:
    """Devuelve dict nombre->[ (s,e), ... ] para saber si estaba en cancha."""
    out = {}
    for (n,_), g in df_pres.groupby(["nombre","rol"], dropna=False):
        ints = [(float(s), float(e)) for s,e in zip(g["start_s"], g["end_s"]) if pd.notna(s) and pd.notna(e) and e> s]
        ints.sort()
        merged=[]
        for s,e in ints:
            if not merged or s>merged[-1][1]:
                merged.append([s,e])
            else:
                merged[-1][1]=max(merged[-1][1], e)
        out.setdefault(n, []).extend([(s,e) for s,e in merged])
    return out

def pr_on_court(name_intervals, t):
    if t is None: return False
    for s,e in name_intervals:
        if s<=t<e: return True
    return False

def pr_procesar(df_raw: pd.DataFrame, df_pres: pd.DataFrame|None, jugador_filter: str|None):
    """
    Devuelve:
      total_acc, perdidas, recupera, porc_perd, porc_recu, df_reg (row,col,zona,jugador,tipo,peso,t0)
    Si jugador_filter está definido, se filtran instancias a las del jugador (y sus minutos en cancha si hay presencias).
    """
    name2ints = pr_merge_minutes(df_pres) if df_pres is not None and not df_pres.empty else {}

    total_acc = np.zeros((PR_N_ROWS, PR_N_COLS), dtype=float)
    perdidas  = np.zeros_like(total_acc)
    recupera  = np.zeros_like(total_acc)

    registros = []

    for _, r in df_raw.iterrows():
        code = r["jugador"]; labels_low = pr_labels_low(r["labels"])
        if jugador_filter:
            # filtrar por nombre (substring case-insensitive)
            nombre, _ = pr_split_name_role(code)
            if not nombre or jugador_filter.lower() not in nombre.lower():
                continue
            # si tenemos intervalos, respetarlos por timestamp
            if name2ints.get(nombre):
                if not pr_on_court(name2ints[nombre], r.get("start")):
                    continue

        pxs, pys = r["pos_x_list"], r["pos_y_list"]
        if not pxs: 
            continue

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
            if p is None: 
                continue
            x,y = pr_transform_xy(*p)
            y = pr_ajustar_ala(y, code)
            rr,cc = pr_zona_from_xy(x,y)
            total_acc[rr,cc]+=1
            registros.append((rr,cc,pr_zona_id(rr,cc), code, labels_low, "OTROS", 1.0, r.get("start")))

    with np.errstate(divide='ignore', invalid='ignore'):
        porc_perd = np.divide(perdidas, total_acc, out=np.zeros_like(perdidas), where=total_acc>0)
        porc_recu = np.divide(recupera, total_acc, out=np.zeros_like(recupera), where=total_acc>0)

    df_reg = pd.DataFrame(registros, columns=["row","col","zona","jugador","labels_low","tipo","peso","t0"])
    return total_acc, perdidas, recupera, porc_perd, porc_recu, df_reg

# ---- Visuales
def pr_draw_pitch_grid(ax):
    dx, dy = PR_ANCHO / PR_N_COLS, PR_ALTO / PR_N_ROWS
    ax.set_facecolor("white")
    ax.plot([0, PR_ANCHO],[0,0], color="black")
    ax.plot([0, PR_ANCHO],[PR_ALTO,PR_ALTO], color="black")
    ax.plot([0,0],[0,PR_ALTO], color="black")
    ax.plot([PR_ANCHO,PR_ANCHO],[0,PR_ALTO], color="black")
    ax.plot([PR_ANCHO/2, PR_ANCHO/2],[0,PR_ALTO], color="black")
    ax.add_patch(Arc((0, PR_ALTO/2), 8, 12, angle=0, theta1=270, theta2=90, color="black"))
    ax.add_patch(Arc((PR_ANCHO, PR_ALTO/2), 8, 12, angle=0, theta1=90, theta2=270, color="black"))
    ax.add_patch(plt.Circle((PR_ANCHO/2, PR_ALTO/2), 4, color="black", fill=False))
    ax.add_patch(plt.Circle((PR_ANCHO/2, PR_ALTO/2), 0.2, color="black"))
    for j in range(PR_N_ROWS):
        for i in range(PR_N_COLS):
            x0, y0 = i*dx, j*dy
            ax.add_patch(Rectangle((x0,y0), dx,dy, linewidth=0.6, edgecolor='gray', facecolor='none'))
            zona = j*PR_N_COLS + i + 1
            ax.text(x0 + dx - 0.4, y0 + dy - 0.4, str(zona), ha='right', va='top', fontsize=9, color='gray')
    ax.set_xlim(0, PR_ANCHO); ax.set_ylim(0, PR_ALTO); ax.axis('off')

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
    """
    Usa el objeto de discover/infer: intenta primero ' - XML NacSport.xml',
    si no, cae a ' - asXML TotalValues.xml'
    """
    if match_obj.get("xml_jugadores") and os.path.isfile(match_obj["xml_jugadores"]):
        return match_obj["xml_jugadores"]
    # fallback dentro de DATA_MINUTOS
    rival = match_obj.get("rival") or ""
    cands = []
    for pat in [
        f"*{rival}*XML NacSport*.xml",
        f"*{rival}*asXML TotalValues*.xml"
    ]:
        cands += glob.glob(os.path.join(DATA_MINUTOS, pat))
    return cands[0] if cands else None

# =========================
# FIN HELPERS
# =========================

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
        # NUEVO: tiempos
        try:
            st = float(inst.findtext("start") or "0")
        except:
            st = 0.0
        try:
            en = float(inst.findtext("end") or "0")
        except:
            en = st

        labels_lc = [nlower(t.text) for t in inst.findall("./label/text")]
        xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
        ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
        x_end, y_end = (xs[-1], ys[-1]) if xs and ys else (None, None)

        out.append({
            "code": code,
            "labels_lc": labels_lc,
            "start": st,
            "end": en,
            "end_xy": (x_end, y_end)
        })
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
    ["📊 Estadísticas de partido", "⏱️ Timeline de Partido", "🔥 Mapas de calor", "🕓 Distribución de minutos","🔗 Red de Pases", "🛡️ Pérdidas y Recuperaciones", "🎯 Mapa de tiros", "🗺️ Mapa 3x3", "⚡ Radar"],
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

elif menu == "⏱️ Timeline de Partido":
    matches = list_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos con patrón: 'Fecha N° - Rival - XML TotalValues.xml'.")
        st.stop()

    sel = st.selectbox("Elegí partido", matches, index=0)
    rival = rival_from_label(sel)

    XML_PATH, _ = infer_paths_for_label(sel)
    if not XML_PATH:
        st.error("No se encontró el XML del partido seleccionado.")
        st.stop()

    # timeline (usa el mismo XML de jugadores/totalvalues)
    tl = build_timeline(XML_PATH)

    ferro_logo = badge_path_for("ferro")
    rival_logo = badge_path_for(rival)

    draw_timeline_panel(
        rival_name=rival,
        tl=tl,
        ferro_logo_path=ferro_logo,
        rival_logo_path=rival_logo
    )

# === Nueva página
elif menu == "🔥 Mapas de calor":
    matches = list_matches()
    if not matches:
        st.warning("No encontré partidos en data/minutos con patrón: 'Fecha N° - Rival - XML TotalValues.xml'.")
        st.stop()

    sel = st.selectbox("Elegí partido", matches, index=0)
    XML_PATH, _ = infer_paths_for_label(sel)
    if not XML_PATH:
        st.error("No se encontró el XML del partido seleccionado.")
        st.stop()

    # Cargar y preparar coords
    df_raw = cargar_datos_nacsport(XML_PATH)
    base   = explode_coords_for_heatmap(df_raw)

    if base.empty:
        st.info("El XML no contiene coordenadas utilizables para mapas de calor.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Por jugador (total)", "Por jugador y rol", "Comparar jugadores por rol"])

    # --- Tab 1: Por jugador (total)
    with tab1:
        jugadores = sorted(base["player_name"].dropna().unique().tolist())
        jugador = st.selectbox("Jugador", jugadores, key="hm_j_total")
        df_j = base[base["player_name"] == jugador]
        # Total = ignorar rol ⇒ no se invierte Y
        df_rot = rotate_coords_for_attack_right(df_j, role=None)
        fig = fig_heatmap(df_rot, f"Mapa de Calor — {jugador} (Total)")
        st.pyplot(fig, use_container_width=True)

    # --- Tab 2: Por jugador y rol
    with tab2:
        jugadores = sorted(base["player_name"].dropna().unique().tolist())
        jugador = st.selectbox("Jugador", jugadores, key="hm_j_rol_player")
        roles    = sorted(base.loc[base["player_name"]==jugador, "role"].dropna().unique().tolist())
        if not roles:
            st.info("Ese jugador no tiene roles etiquetados en este partido.")
        else:
            rol = st.selectbox("Rol", roles, key="hm_j_rol_role")
            df_jr = base[(base["player_name"]==jugador) & (base["role"]==rol)]
            df_rot = rotate_coords_for_attack_right(df_jr, role=rol)
            fig = fig_heatmap(df_rot, f"Mapa de Calor — {jugador} ({rol})")
            st.pyplot(fig, use_container_width=True)

    # --- Tab 3: Comparar jugadores por rol
    with tab3:
        # rol de referencia
        roles_all = sorted(base["role"].dropna().unique().tolist())
        if not roles_all:
            st.info("Este partido no tiene roles etiquetados para comparar.")
        else:
            rol_cmp = st.selectbox("Rol", roles_all, key="hm_cmp_role")
            # elegimos múltiples jugadores que hayan jugado ese rol
            jugadores_rol = sorted(base.loc[base["role"]==rol_cmp, "player_name"].dropna().unique().tolist())
            sel_players = st.multiselect("Jugadores", jugadores_rol, default=jugadores_rol[:2], key="hm_cmp_players")

            if not sel_players:
                st.info("Seleccioná al menos un jugador.")
            else:
                # Layout flexible: 2 por fila
                n = len(sel_players)
                cols = st.columns(2) if n > 1 else [st]
                for i, name in enumerate(sel_players):
                    df_jr = base[(base["player_name"]==name) & (base["role"]==rol_cmp)]
                    df_rot = rotate_coords_for_attack_right(df_jr, role=rol_cmp)
                    fig = fig_heatmap(df_rot, f"{name} ({rol_cmp})")
                    with cols[i % 2]:
                        st.pyplot(fig, use_container_width=True)

elif menu == "🕓 Distribución de minutos":
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
        st.warning("No encontré el XML TotalValues del partido seleccionado.")
        st.stop()

    # 1) Cargar presencias desde XML TotalValues (jugador (rol) únicamente) y calcular minutos
    df_pres = cargar_minutos_desde_xml_totalvalues(m["xml_players"])
    df_por_rol, df_por_jugador = minutos_por_presencia(df_pres)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Tabla — Minutos por jugador (rol)**")
        st.dataframe(df_por_rol[["nombre","rol","mmss","minutos","n_tramos"]], use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV (por rol)",
            df_por_rol.to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}_minutos_por_rol.csv",
            mime="text/csv"
        )
    with cB:
        st.markdown("**Tabla — Minutos por jugador (total)**")
        st.dataframe(df_por_jugador[["nombre","mmss","minutos","n_tramos"]], use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV (total por jugador)",
            df_por_jugador.to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}_minutos_por_jugador.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # 2) Gráfico A — por jugador (rol)
    if not df_por_rol.empty:
        labels_A = [f"{n} ({r}) — n={t}" for n, r, t in zip(df_por_rol["nombre"], df_por_rol["rol"], df_por_rol["n_tramos"])]
        vals_A   = df_por_rol["segundos"].tolist()
        figA = fig_barh_minutos(labels_A, vals_A, "Minutos por jugador (por rol) — partido")
        st.pyplot(figA, use_container_width=True)

    # 3) Gráfico B — total por jugador
    if not df_por_jugador.empty:
        labels_B = [f"{n} — n={t}" for n, t in zip(df_por_jugador["nombre"], df_por_jugador["n_tramos"])]
        vals_B   = df_por_jugador["segundos"].tolist()
        figB = fig_barh_minutos(labels_B, vals_B, "Minutos por jugador (TOTAL) — partido")
        st.pyplot(figB, use_container_width=True)

    # 4) Gráfico C — por cada rol (ordenado desc.)
    st.markdown("### Por rol (ordenado de mayor a menor)")
    if df_por_rol.empty:
        st.info("Sin datos de roles para este partido.")
    else:
        roles = sorted(df_por_rol["rol"].dropna().unique().tolist())
        selected_roles = st.multiselect("Filtrar roles", roles, default=roles)
        gcols = st.columns(2) if len(selected_roles) > 1 else [st]
        i = 0
        for rol in selected_roles:
            g = df_por_rol[df_por_rol["rol"] == rol].sort_values("segundos", ascending=False)
            labels_R = [f"{n} — n={t}" for n, t in zip(g["nombre"], g["n_tramos"])]
            vals_R   = g["segundos"].tolist()
            figR = fig_barh_minutos(labels_R, vals_R, f"Minutos por rol — {rol}")
            with gcols[i % len(gcols)]:
                st.pyplot(figR, use_container_width=True)
            i += 1

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
        st.warning("No encontré el XML TotalValues del partido seleccionado.")
        st.stop()

    # cargar y filtrar
    df_all = cargar_datos_nacsport(m["xml_players"])

    # (opcional) selector de roles a mostrar en el grafo: si lo querés, descomenta:
    # roles_detectados = sorted(set(
    #     r.split("(")[1].replace(")","").strip()
    #     for r in df_all["jugador"] if "(" in str(r) and ")" in str(r)
    # ))
    # roles_sel = st.multiselect("Filtrar roles (opcional)", roles_detectados, default=roles_detectados)
    # if roles_sel:
    #     df_all = df_all[df_all["jugador"].apply(lambda s: any(f"({r})" in str(s) for r in roles_sel))]

    fig = red_de_pases_por_rol(df_all)
    st.pyplot(fig, use_container_width=True)

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
    xml_eq  = m.get("xml_equipo")  # puede ser None; funciona igual (sin minutos de cancha)

    if not xml_jug or not os.path.isfile(xml_jug):
        st.error("No encontré XML NacSport/TotalValues con eventos de jugadores para este partido.")
        st.stop()

    # Cargar
    df_raw  = pr_cargar_datos(xml_jug)
    df_pres = pr_cargar_presencias_equipo(xml_eq) if xml_eq else pd.DataFrame(columns=["nombre","rol","start_s","end_s"])

    st.caption(f"Usando eventos de: `{os.path.basename(xml_jug)}`" + (f" y presencias: `{os.path.basename(xml_eq)}`" if xml_eq else " (sin XML de equipo)"))

    # Modo de análisis
    modo = st.radio("Modo", ["Equipo", "Jugador"], horizontal=True)
    jugador_sel = None
    if modo == "Jugador":
        # lista de nombres a partir de códigos "Nombre (Rol)"
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
        return {"total":tot,"perd":perd,"recu":recu,"pperd":pperd,"precu":precu,"reg":dreg,"res":dfres}

    if not acum:
        R = _procesar_un_partido(m)
    else:
        # acumular todos
        agg = None
        for mi in matches:
            Ri = _procesar_un_partido(mi)
            if Ri is None: 
                continue
            if agg is None:
                agg = Ri
            else:
                for k in ["total","perd","recu","pperd","precu"]:
                    agg[k] += Ri[k]  # acumulamos contadores; % los recalculamos luego
                agg["reg"] = pd.concat([agg["reg"], Ri["reg"]], ignore_index=True)
        if agg is None:
            st.error("No se pudo acumular: no hay partidos con XML válido.")
            st.stop()
        # Recalcular porcentajes con los acumulados
        with np.errstate(divide='ignore', invalid='ignore'):
            pperd = np.divide(agg["perd"], agg["total"], out=np.zeros_like(agg["perd"]), where=agg["total"]>0)
            precu = np.divide(agg["recu"], agg["total"], out=np.zeros_like(agg["recu"]), where=agg["total"]>0)
        agg["pperd"], agg["precu"] = pperd, precu
        agg["res"] = pr_resumen_df(agg["total"], agg["perd"], agg["recu"], pperd, precu)
        R = agg

    if R is None:
        st.error("No hay datos para procesar.")
        st.stop()

    # Tablas
    st.markdown("#### Resumen por zona")
    st.dataframe(R["res"][["zona","total_acciones","%_recuperaciones_sobre_total","%_perdidas_sobre_total"]]
                 .rename(columns={
                     "%_recuperaciones_sobre_total":"% RECUP.",
                     "%_perdidas_sobre_total":"% PÉRD."
                 }), use_container_width=True)

    # Heatmaps
    M_tot  = R["total"]
    figR   = pr_heatmap(R["precu"], M_tot, f"{'Equipo' if modo=='Equipo' else jugador_sel} — % RECUPERACIONES", good_high=True)
    figP   = pr_heatmap(R["pperd"], M_tot, f"{'Equipo' if modo=='Equipo' else jugador_sel} — % PÉRDIDAS",       good_high=False)
    st.pyplot(figR, use_container_width=True)
    st.pyplot(figP, use_container_width=True)

    # Barras ordenadas
    figBR = pr_bars(R["res"], "%_recuperaciones_sobre_total", f"{'Equipo' if modo=='Equipo' else jugador_sel} — Ranking Zonas por % RECUP.")
    figBP = pr_bars(R["res"], "%_perdidas_sobre_total",       f"{'Equipo' if modo=='Equipo' else jugador_sel} — Ranking Zonas por % PÉRD.")
    st.pyplot(figBR, use_container_width=True)
    st.pyplot(figBP, use_container_width=True)

elif menu == "🎯 Mapa de tiros":
    st.subheader("Mapa de tiros — por partido / jugador / rol")

    # --------------------------
    # Resolver partido y XML
    # --------------------------
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

    # Preferimos NacSport; si no hubiera, intentamos con TotalValues (se filtra por 'tiro' igualmente)
    xml_path = m.get("xml_nacsport") or m.get("xml_jugadores") or m.get("xml_totalvalues")
    if not xml_path or not os.path.isfile(xml_path):
        st.warning("No encontré XML NacSport/TotalValues para este partido.")
        st.stop()

    # --------------------------
    # Helpers específicos
    # --------------------------
    import re, xml.etree.ElementTree as ET
    from collections import Counter
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Arc

    ANCHO, ALTO = 35.0, 20.0
    N_COLS, N_ROWS = 3, 3
    FLIP_TO_RIGHT  = True   # ataque hacia la derecha
    FLIP_VERTICAL  = True   # espejo vertical (arriba/abajo)
    GOAL_PULL      = 0.60   # “tirón” suave hacia el arco derecho (visual)

    def ntext(s):
        import unicodedata
        if s is None: return ""
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return s.strip()
    def nlower(s): return ntext(s).lower()

    def parse_instances_generic(xml_path):
        """Lee instancias (NacSport o TotalValues). Devuelve lista y max_x/max_y para escalar."""
        root = ET.parse(xml_path).getroot()
        out, all_x, all_y = [], [], []
        for inst in root.findall(".//instance"):
            code   = ntext(inst.findtext("code"))
            labels = [nlower(t.text) for t in inst.findall("./label/text")]
            xs = [int(x.text) for x in inst.findall("./pos_x") if (x.text or "").isdigit()]
            ys = [int(y.text) for y in inst.findall("./pos_y") if (y.text or "").isdigit()]
            if xs and ys: all_x += xs; all_y += ys
            out.append({"code": code, "labels": labels, "xs": xs, "ys": ys})
        max_x = float(max(all_x) if all_x else 19)
        max_y = float(max(all_y) if all_y else 34)
        return out, max_x, max_y

    # ¿Es tiro?
    def is_shot(ev):
        s = nlower(ev["code"])
        patt = re.compile(r"\b(tiro|remate)\b")
        return bool(patt.search(s) or any(patt.search(l or "") for l in ev["labels"]))

    # Extraer jugador y rol desde "Nombre (Rol)". Si no, intentamos detectar dentro de labels.
    _NAME_ROLE_RE = re.compile(r"^\s*([^(]+?)\s*\(([^)]+)\)\s*$")
    def extract_name_role(ev):
        m = _NAME_ROLE_RE.match(ev["code"])
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # fallback: buscar "(Rol)" en labels
        for l in ev["labels"]:
            if not l: continue
            mx = re.search(r"\(([^)]+)\)", l)
            if mx:
                return None, mx.group(1).strip()
        return None, None

    # Clasificación del resultado (estricta)
    KEYS = {
        "gol":        re.compile(r"\bgol\b"),
        "ataj":       re.compile(r"\bataj"),
        "al_arco":    re.compile(r"\btiro\s*al\s*arco\b"),
        "bloqueado":  re.compile(r"\bbloquead"),
        "desviado":   re.compile(r"\bdesviad"),
        "errado":     re.compile(r"\berrad"),
        "pifia":      re.compile(r"\bpifi"),
    }
    def has(ev, key):
        patt = KEYS[key]
        if patt.search(nlower(ev["code"])): return True
        return any(patt.search(l or "") for l in ev["labels"])
    def shot_result_strict(ev):
        if has(ev, "gol"):                         return "Gol"
        if has(ev, "ataj") or has(ev, "al_arco"):  return "Tiro Atajado"
        if has(ev, "bloqueado"):                   return "Tiro Bloqueado"
        if has(ev, "desviado"):                    return "Tiro Desviado"
        if has(ev, "errado") or has(ev, "pifia"):  return "Tiro Errado - Pifia"
        return "Sin clasificar"

    # Característica del ORIGEN
    CHAR_PATTS = [
        ("de Corner (desde Banda)", re.compile(r"corner\s*\(desde\s*banda\)")),
        ("de Corner (centro)",      re.compile(r"corner\s*\(centro\)")),
        ("de Jugada (centro)",      re.compile(r"jugada\s*\(centro\)")),
        ("de Tiro Libre",           re.compile(r"tiro\s*libre")),
        ("de Rebote",               re.compile(r"\brebote\b")),
        ("de Lateral",              re.compile(r"\blateral\b")),
        ("de Jugada",               re.compile(r"\bde\s*jugada\b")),
    ]
    def shot_characteristic(ev):
        s = nlower(ev["code"]) + " " + " ".join(ev["labels"])
        for name, patt in CHAR_PATTS:
            if patt.search(s): return name
        return "de Jugada"

    # Mapeo de coords al campo
    def map_raw_to_pitch(x_raw, y_raw, max_x, max_y, flip=True, pull=0.0, flip_v=False):
        x = (y_raw / max_y) * ANCHO
        y = (x_raw / max_x) * ALTO
        if flip:   x = ANCHO - x
        if flip_v: y = ALTO - y
        if pull and 0.0 < pull < 1.0:
            x = x + pull * (ANCHO - x)
        x = float(np.clip(x, 0.0, ANCHO)); y = float(np.clip(y, 0.0, ALTO))
        return x, y

    def draw_futsal_pitch_grid(ax):
        dx, dy = ANCHO / N_COLS, ALTO / N_ROWS
        ax.set_facecolor("white")
        ax.plot([0, ANCHO], [0, 0], color="black")
        ax.plot([0, ANCHO], [ALTO, ALTO], color="black")
        ax.plot([0, 0], [0, ALTO], color="black")
        ax.plot([ANCHO, ANCHO], [0, ALTO], color="black")
        ax.plot([ANCHO/2, ANCHO/2], [0, ALTO], color="black")
        ax.add_patch(Arc((0, ALTO/2), 8, 12, angle=0, theta1=270, theta2=90, color="black"))
        ax.add_patch(Arc((ANCHO, ALTO/2), 8, 12, angle=0, theta1=90, theta2=270, color="black"))
        ax.add_patch(plt.Circle((ANCHO/2, ALTO/2), 4, color="black", fill=False))
        ax.add_patch(plt.Circle((ANCHO/2, ALTO/2), 0.2, color="black"))
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
            # Solo mis jugadores: "Nombre (Rol)"; ignoramos "Categoría - Equipo Rival" y tiempos
            name, role = extract_name_role(ev)
            code_norm = nlower(ev["code"])
            if any(code_norm.startswith(prefix) for prefix in (
                "categoria - equipo rival", "categoría - equipo rival",
                "tiempo posecion ferro", "tiempo posesion ferro",
                "tiempo posecion rival", "tiempo posesion rival",
                "tiempo no jugado"
            )):
                continue
            # si no hay nombre (no está "Nombre (Rol)") lo descartamos
            if not name and not role:
                continue
            if not (ev["xs"] and ev["ys"]):
                continue

            coords = [map_raw_to_pitch(xr, yr, max_x, max_y,
                                       flip=FLIP_TO_RIGHT, pull=GOAL_PULL, flip_v=FLIP_VERTICAL)
                      for xr, yr in zip(ev["xs"], ev["ys"])]

            # origen = punto más lejano al arco derecho (x más chico tras flip)
            origin = coords[0] if len(coords)==1 else coords[int(np.argmin([c[0] for c in coords]))]

            shots.append({
                "x": origin[0], "y": origin[1],
                "result": shot_result_strict(ev),
                "char":   shot_characteristic(ev),
                "player": name or "", "role": role or ""
            })
        return shots

    # --------------------------
    # Cargar y preparar filtros
    # --------------------------
    shots = collect_shots(xml_path)
    if not shots:
        st.info("No se detectaron tiros para este partido.")
        st.stop()

    dfS = pd.DataFrame(shots)
    jugadores = sorted([j for j in dfS["player"].unique() if j])
    roles     = sorted([r for r in dfS["role"].unique() if r])
    chars     = sorted(dfS["char"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        pick_players = st.multiselect("Jugadores", jugadores, default=jugadores)
    with c2:
        pick_roles   = st.multiselect("Roles", roles, default=roles)
    with c3:
        pick_chars   = st.multiselect("Característica (origen)", chars, default=chars)

    # Filtro
    f = dfS.copy()
    if pick_players:
        f = f[f["player"].isin(pick_players)]
    if pick_roles:
        f = f[f["role"].isin(pick_roles)]
    if pick_chars:
        f = f[f["char"].isin(pick_chars)]

    # --------------------------
    # Plot de tiros
    # --------------------------
    order = ["Gol","Tiro Atajado","Tiro Bloqueado","Tiro Desviado","Tiro Errado - Pifia","Sin clasificar"]
    COLORS = {
        "Gol":                "#FFD54F",
        "Tiro Atajado":       "#FFFFFF",
        "Tiro Bloqueado":     "#FF5252",
        "Tiro Desviado":      "#FF7043",
        "Tiro Errado - Pifia":"#6B6F76",
        "Sin clasificar":     "#BDBDBD",
    }

    plt.close("all")
    fig = plt.figure(figsize=(10.5, 7))
    ax  = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    draw_futsal_pitch_grid(ax)

    for res in order:
        pts = f.loc[f["result"] == res, ["x","y"]].to_numpy()
        if pts.size == 0: continue
        xs, ys = pts[:,0], pts[:,1]
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

    sub_players = ", ".join(pick_players) if pick_players else "Todos"
    sub_roles   = ", ".join(pick_roles)   if pick_roles   else "Todos"
    sub_chars   = ", ".join(pick_chars)   if pick_chars   else "Todas"
    ax.set_title(
        f"SHOTS — Origen (punto más lejano) | Jugadores: {sub_players} | Roles: {sub_roles}\n"
        f"Característica: {sub_chars}",
        fontsize=13, pad=6, weight="bold"
    )
    ax.legend(loc="upper left", frameon=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --------------------------
    # Tablas de conteo + descarga
    # --------------------------
    from collections import defaultdict
    want_order = ["Gol","Tiro Atajado","Tiro Bloqueado","Tiro Desviado","Tiro Errado - Pifia"]
    c_res = Counter(r for r in f["result"] if r != "Sin clasificar")
    results_counts = {k: int(c_res.get(k, 0)) for k in want_order}
    c_char = Counter(f["char"])
    char_counts = dict(sorted(c_char.items(), key=lambda kv: (-kv[1], kv[0])))

    df_results = (pd.DataFrame([
        {"Categoría": k, "Conteo": v, "%": f"{(v/sum(results_counts.values())*100) if sum(results_counts.values()) else 0:.1f}%"}
        for k, v in results_counts.items()
    ] + ([{"Categoría":"TOTAL","Conteo":sum(results_counts.values()), "%":"100.0%" if sum(results_counts.values()) else "0.0%"}]))
    )
    df_chars = (pd.DataFrame([
        {"Categoría": k, "Conteo": v, "%": f"{(v/sum(c_char.values())*100) if sum(c_char.values()) else 0:.1f}%"}
        for k, v in char_counts.items()
    ] + ([{"Categoría":"TOTAL","Conteo":sum(c_char.values()), "%":"100.0%" if sum(c_char.values()) else "0.0%"}]))
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

else:
    st.info("Las demás secciones se irán conectando con tus notebooks en los próximos pasos.")
