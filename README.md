# ⚽ InsightFutsal

**InsightFutsal** es una aplicación interactiva desarrollada en **Streamlit** para el análisis de partidos y jugadores de futsal.  
Integra los registros obtenidos en **NacSport**, procesados en **Python**, y genera métricas avanzadas y visualizaciones intuitivas que ayudan a **entrenadores, jugadores y clubes** a tomar decisiones basadas en datos.

---

## 🚀 Funcionalidades principales

- ⏱️ **Minutos jugados por rol y jugador**: desglose y comparativa por posiciones.  
- 🎯 **Tiros por equipo y jugador**: volumen, efectividad y ubicación en cancha.  
- 🗺️ **Mapas de calor (3x3 y 4x4)**: distribución espacial de acciones, recuperaciones y pérdidas.  
- 🔄 **Pérdidas y recuperaciones**: análisis por jugador, equipo y zonas del campo.  
- 🤝 **Red de pases por partido**: conexiones, flujo de juego y jugadores más influyentes.  
- 📈 **Estadísticas del partido**: métricas agregadas para cada equipo (tiros, xG, posesión, etc.).  
- 🕒 **Timeline del partido**: evolución temporal de tiros, goles y eventos clave.  
- 🛡️ **Radar individual y comparativo**: perfil de rendimiento ofensivo, defensivo o mixto.  
- 📊 **Tabla de posiciones y ranking Elo**: evolución fecha a fecha considerando rendimiento y rivales.  

---

## 🛠️ Tecnologías utilizadas

- [Python 3.10+](https://www.python.org/)  
- [Streamlit](https://streamlit.io/)  
- [Pandas](https://pandas.pydata.org/)  
- [Matplotlib](https://matplotlib.org/) / [Plotly](https://plotly.com/)  
- [NetworkX](https://networkx.org/) (redes de pase)  
- [Scikit-learn](https://scikit-learn.org/)  
- Archivos de entrada: **XML / XLSX** exportados desde **NacSport**  

---

## 📂 Estructura del proyecto
📦 InsightFutsal

┣ 📂 data/ # Archivos de entrada (XML, XLSX)

┣ 📂 src/ # Scripts de procesamiento y análisis

┣ 📂 notebooks/ # Desarrollo en Google Colab / Jupyter

┣ 📂 visuals/ # Gráficos y recursos generados

┣ app.py # App principal de Streamlit

┣ requirements.txt # Librerías necesarias

┗ README.md


---

## ▶️ Cómo ejecutar la app

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/insightfutsal.git
   cd insightfutsal
  
2. Instalar dependencias:
  ```bash
  pip install -r requirements.txt   
  ``` 
3. Ejecutar la aplicación:
  ```bash
  streamlit run app.py
  ```

🎯 Objetivo del proyecto

InsightFutsal busca democratizar el análisis avanzado en el futsal y fútbol formativo, brindando a clubes amateurs, entrenadores y jugadores jóvenes una herramienta simple y accesible que convierta los datos de partido en información práctica para optimizar el rendimiento, detectar talento y mejorar la toma de decisiones.

👤 Autor

Desarrollado por Cristian Dieguez
📧 Contacto: [LinkedIn](https://www.linkedin.com/in/cristiandieguez/)
