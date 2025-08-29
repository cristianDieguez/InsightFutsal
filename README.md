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

📦 InsightFutsal
├─ 📂 data/                 # Entradas

│  ├─ 📂 minutos/           # XML “TotalValues” (por partido)

│  └─ 📂 matrix/            # XLSX “Matrix” (por partido)

├─ 📂 src/                  # Procesamiento y helpers

├─ 📂 notebooks/            # Colab/Jupyter (exploración)

├─ 📂 visuals/              # Gráficos exportados

├─ app.py                   # App principal Streamlit

├─ requirements.txt         # Dependencias

└─ README.md

Formato esperado de archivos (recomendado):

data/minutos/Fecha N - Rival - XML TotalValues.xml

data/matrix/Fecha N - Rival - Matrix.xlsx

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

🧭 Uso rápido

Elegí el menú (Minutos, Tiros, Mapas, Red de pases, Radar, etc.).
Fijá el alcance (partido único o todos los partidos).
En Radar, podés comparar “Jugador total”, “Por rol” o “Jugador y rol”.
Métricas % se grafican tal cual (0–100).
Métricas absolutas se normalizan por el máximo global del grupo (a 40’ por partido), y los anillos muestran los valores de referencia coherentes con la tabla.

🧩 Entradas “Matrix” (agrupaciones usadas)

Pases: Corto/Progresivo × Frontal/Lateral (+ Completado/OK)
Centros (+ rematados)
Tiros: intentos, al arco, bloqueados, desviados, goles
Regates: conseguidos/no conseguidos × (mantiene/pierde)
Pivot: aguanta/gira
Presión: presiona/presionado
Faltas: hechas/recibidas
Recuperaciones/Perdidas (por causa)
1v1: ganado/perdido
Asistencia / Pase Clave, Conducción, Despeje, Gol
El README refleja las métricas derivadas: % Regates Exitosos, % Duelos Ganados, Tiros - % al arco, Tiros - % Goles/TA, % Recuperaciones, % Acciones Positivas, etc.

🎯 Objetivo del proyecto

InsightFutsal busca democratizar el análisis avanzado en el futsal y fútbol formativo, brindando a clubes amateurs, entrenadores y jugadores jóvenes una herramienta simple y accesible que convierta los datos de partido en información práctica para optimizar el rendimiento, detectar talento y mejorar la toma de decisiones.

👤 Autor

Desarrollado por Cristian Dieguez
📧 Contacto: [LinkedIn](https://www.linkedin.com/in/cristiandieguez/)
