import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(
    page_title="TFM Ferro Futsal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas base (ajusta si cambian)
DATA_PATH = "data/minutos"
IMAGES_PATH = "images/equipos"
BANNER_PATH = "images/banner"

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def cargar_xml_totalvalues(path_xml):
    tree = ET.parse(path_xml)
    root = tree.getroot()
    data = []
    for child in root.findall(".//Value"):
        fila = {k: child.get(k) for k in child.keys()}
        data.append(fila)
    return pd.DataFrame(data)

def graficar_estadisticas(df_stats, rival, fecha):
    fig, ax = plt.subplots(figsize=(10,5))
    df_stats.plot(kind="bar", ax=ax)
    ax.set_title(f"Estadísticas vs {rival} (Fecha {fecha})")
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("📊 Panel Futsal Ferro")
seccion = st.sidebar.radio("Elegí sección:", [
    "🏠 Inicio",
    "📈 Estadísticas de Partido",
    "🎯 Tiros",
    "🔥 Mapa de Calor",
    "🕒 Timeline",
    "🔗 Red de Pases",
    "⚽ Pérdidas y Recuperaciones",
    "📊 Elo Ranking"
])

# ==========================
# INICIO
# ==========================
if seccion == "🏠 Inicio":
    st.image(os.path.join(BANNER_PATH, "banner.png"))
    st.title("TFM Ferro Futsal")
    st.markdown("Bienvenido al panel de análisis de Ferro Carril Oeste - Promocionales 2016")

# ==========================
# ESTADÍSTICAS PARTIDO
# ==========================
elif seccion == "📈 Estadísticas de Partido":
    st.header("📈 Estadísticas de Partido")
    
    archivos_xml = sorted([f for f in os.listdir(DATA_PATH) if f.endswith("XML TotalValues.xml")])
    
    if not archivos_xml:
        st.warning("No se encontraron archivos en `data/minutos` con formato `Fecha N° - Rival - XML TotalValues.xml`")
    else:
        archivo_sel = st.selectbox("Elegí un partido:", archivos_xml)
        path_xml = os.path.join(DATA_PATH, archivo_sel)

        # Parseo nombre
        try:
            fecha = archivo_sel.split(" - ")[0].replace("Fecha ", "")
            rival = archivo_sel.split(" - ")[1]
        except:
            fecha, rival = "?", "?"

        df_stats = cargar_xml_totalvalues(path_xml)

        if df_stats.empty:
            st.error("⚠️ No se pudieron extraer estadísticas del XML.")
        else:
            st.subheader(f"📊 Partido vs {rival} (Fecha {fecha})")

            # Preview tabla
            st.dataframe(df_stats.head())

            # Gráfico
            st.markdown("### Gráfico de Estadísticas")
            try:
                df_plot = df_stats.set_index("label")[["count"]].astype(float)
                graficar_estadisticas(df_plot, rival, fecha)
            except Exception as e:
                st.error(f"No se pudo graficar: {e}")

# ==========================
# (otras secciones: Tiros, Mapa de Calor, etc.)
# ==========================
elif seccion == "🎯 Tiros":
    st.header("🎯 Tiros por equipo y jugador")
    st.info("👉 Esta sección se adaptará del notebook `Tiros por equipo y jugador.ipynb`.")

elif seccion == "🔥 Mapa de Calor":
    st.header("🔥 Mapa de Calor por jugador y rol")
    st.info("👉 Esta sección se adaptará del notebook `Mapa de Calor por Jugador y Rol.ipynb`.")

elif seccion == "🕒 Timeline":
    st.header("🕒 Timeline del partido")
    st.info("👉 Esta sección se adaptará del notebook `Timeline Partido.ipynb`.")

elif seccion == "🔗 Red de Pases":
    st.header("🔗 Red de pases por partido")
    st.info("👉 Esta sección se adaptará del notebook `Red de pases por partido.ipynb`.")

elif seccion == "⚽ Pérdidas y Recuperaciones":
    st.header("⚽ Pérdidas y Recuperaciones por equipo y jugador")
    st.info("👉 Esta sección se adaptará de los notebooks de recuperaciones y pérdidas.")

elif seccion == "📊 Elo Ranking":
    st.header("📊 Tabla de Posiciones & Elo Ranking")
    st.info("👉 Esta sección se adaptará del notebook `Tabla Posiciones & Elo Ranking por Fecha.ipynb`.")
