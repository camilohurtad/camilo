import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="Dashboard Afectaciones Digitales – Sismo 10 Ago 2026",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0F2C59 0%, #1D5B79 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { color: white; font-size: 1.75rem; margin: 0; }
    .main-header p { color: #E0ECF8; font-size: 0.95rem; margin: 0.3rem 0 0 0; }
</style>
""", unsafe_allow_html=True)

def buscar_columna(df, patrones):
    for col in df.columns:
        col_low = col.lower()
        if all(p in col_low for p in patrones):
            return col
    return None

@st.cache_data(ttl=3600)
def cargar_y_limpiar_datos(archivo_subido=None):
    if archivo_subido is not None:
        df_raw = pd.read_csv(archivo_subido)
    else:
        url_api = "https://www.datos.gov.co/resource/v3tn-7h59.csv?$limit=1000"
        df_raw = pd.read_csv(url_api)

    df = df_raw.copy()
    cols_pii = [
        c for c in df.columns
        if any(k in c.lower() for k in ["correo", "nombre", "celular", "telefono", "teléfono"])
        and "razon" not in c.lower() and "entidad" not in c.lower()
    ]
    df.drop(columns=cols_pii, inplace=True, errors="ignore")

    mapa = {
        buscar_columna(df, ["departamento"]): "departamento",
        buscar_columna(df, ["municipio"]): "municipio",
        buscar_columna(df, ["tipo", "entidad"]): "tipo_entidad",
        buscar_columna(df, ["razon"]) or buscar_columna(df, ["entidad"]): "nombre_entidad",
        buscar_columna(df, ["nivel", "general"]): "nivel_afectacion",
        buscar_columna(df, ["situaci", "tic"]) or buscar_columna(df, ["riesgo"]): "riesgo_continuidad_tic",
        buscar_columna(df, ["sistemas", "informaci"]): "sistemas_afectados",
        buscar_columna(df, ["estado", "general"]): "estado_sistemas",
    }
    mapa = {k: v for k, v in mapa.items() if k is not None}
    df.rename(columns=mapa, inplace=True)

    cols_cat = ["departamento", "municipio", "tipo_entidad", "nivel_afectacion", "riesgo_continuidad_tic", "estado_sistemas"]
    for c in cols_cat:
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                .fillna("Sin información")
            )
    if "departamento" in df.columns:
        df["departamento"] = df["departamento"].str.title().str.replace(" Del ", " del ")
    if "municipio" in df.columns:
        df["municipio"] = df["municipio"].str.title()
    if "riesgo_continuidad_tic" in df.columns:
        df["riesgo_continuidad_tic"] = df["riesgo_continuidad_tic"].replace({"Si": "Sí", "SI": "Sí", "NO": "No"})
    if "sistemas_afectados" in df.columns:
        df["sistemas_afectados"] = df["sistemas_afectados"].fillna("Ninguno / Sin reporte")

    df.drop_duplicates(inplace=True)
    return df

st.markdown("""
<div class="main-header">
    <h1>Diagnóstico de Afectaciones en Capacidades Digitales – Sismo 10 de agosto de 2026</h1>
    <p>Fuente: Ministerio TIC – Portal Nacional de Datos Abiertos Colombia (ID: v3tn-7h59) | Electiva III – Ciencia de Datos</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Filtros y Fuente de Datos")
archivo_csv = st.sidebar.file_uploader("Subir CSV alternativo (opcional)", type=["csv"])

try:
    df = cargar_y_limpiar_datos(archivo_csv)
except Exception as e:
    st.error(f"No se pudo cargar la base desde datos.gov.co ({e}). Sube el archivo CSV manualmente en el panel izquierdo.")
    st.stop()

deptos_disp = sorted(df["departamento"].unique())
sel_deptos = st.sidebar.multiselect("Departamento", deptos_disp, default=deptos_disp)

df_temp = df[df["departamento"].isin(sel_deptos)]
munis_disp = sorted(df_temp["municipio"].unique())
sel_munis = st.sidebar.multiselect("Municipio", munis_disp, default=munis_disp)

tipos_disp = sorted(df_temp["tipo_entidad"].unique())
sel_tipos = st.sidebar.multiselect("Tipo de entidad", tipos_disp, default=tipos_disp)

df_filtrado = df[
    (df["departamento"].isin(sel_deptos)) &
    (df["municipio"].isin(sel_munis)) &
    (df["tipo_entidad"].isin(sel_tipos))
]

st.sidebar.markdown("---")
st.sidebar.download_button(
    label="Descargar base limpia filtrada (.csv)",
    data=df_filtrado.to_csv(index=False, encoding="utf-8-sig"),
    file_name="base_limpia_terremoto.csv",
    mime="text/csv"
)

n_total = len(df_filtrado)
n_munis = df_filtrado["municipio"].nunique() if n_total > 0 else 0
n_riesgo = (df_filtrado["riesgo_continuidad_tic"] == "Sí").sum() if n_total > 0 else 0
pct_riesgo = (n_riesgo / n_total * 100) if n_total > 0 else 0.0
n_alta = df_filtrado["nivel_afectacion"].str.contains("Alta|Crítica|Grave|Total", case=False, na=False).sum() if n_total > 0 else 0
pct_alta = (n_alta / n_total * 100) if n_total > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entidades Reportantes", f"{n_total}", help="Total de entidades públicas y educativas en la selección")
c2.metric("Municipios Cubiertos", f"{n_munis}", help="Municipios únicos con reporte")
c3.metric("Riesgo Continuidad TIC", f"{pct_riesgo:.1f}%", f"{n_riesgo} entidades", delta_color="inverse")
c4.metric("Afectación Alta / Crítica", f"{pct_alta:.1f}%", f"{n_alta} entidades", delta_color="inverse")

st.markdown("---")

if n_total == 0:
    st.warning("No hay registros para la combinación de filtros seleccionada.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Panorama Territorial y Sectorial",
    "2. Sistemas Críticos y Estado Operativo",
    "3. Simulador Predictivo (Valor Agregado)",
    "4. Tabla de Datos Depurados"
])

with tab1:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_depto = px.histogram(
            df_filtrado,
            x="departamento",
            color="nivel_afectacion",
            barmode="group",
            title="Nivel general de afectación digital por departamento",
            labels={"departamento": "Departamento", "nivel_afectacion": "Nivel de afectación", "count": "Entidades"},
            color_discrete_sequence=["#0F2C59", "#1D5B79", "#E59838", "#C0392B", "#7F8C8D"],
            template="plotly_white"
        )
        fig_depto.update_layout(yaxis_title="Número de entidades")
        st.plotly_chart(fig_depto, use_container_width=True)

    with col_g2:
        fig_tipo = px.histogram(
            df_filtrado,
            y="tipo_entidad",
            color="riesgo_continuidad_tic",
            barnorm="percent",
            title="Proporción de riesgo en servicios esenciales por tipo de entidad (%)",
            labels={"tipo_entidad": "Tipo de entidad", "riesgo_continuidad_tic": "¿Riesgo TIC?"},
            color_discrete_map={"Sí": "#C0392B", "No": "#1D5B79", "Sin información": "#BDC3C7"},
            template="plotly_white"
        )
        fig_tipo.update_layout(xaxis_title="Porcentaje (%)")
        st.plotly_chart(fig_tipo, use_container_width=True)

with tab2:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sistemas_series = (
            df_filtrado["sistemas_afectados"]
            .dropna().str.split(r"[;,]").explode().str.strip()
        )
        sistemas_series = sistemas_series[~sistemas_series.isin(["", "Ninguno / Sin reporte", "Sin información"])]
        if len(sistemas_series) > 0:
            t_sist = sistemas_series.value_counts().head(10).reset_index()
            t_sist.columns = ["Sistema / Servicio", "Reportes"]
            fig_sist = px.bar(
                t_sist.sort_values("Reportes"),
                x="Reportes",
                y="Sistema / Servicio",
                orientation="h",
                text="Reportes",
                title="Top 10 sistemas o servicios digitales afectados",
                color_discrete_sequence=["#0F2C59"],
                template="plotly_white"
            )
            fig_sist.update_traces(textposition="outside")
            st.plotly_chart(fig_sist, use_container_width=True)
        else:
            st.info("No se registran sistemas específicos afectados en la selección actual.")

    with col_s2:
        t_est = df_filtrado["estado_sistemas"].value_counts().reset_index()
        t_est.columns = ["Estado Operativo", "Entidades"]
        fig_est = px.pie(
            t_est,
            names="Estado Operativo",
            values="Entidades",
            hole=0.45,
            title="Estado general de los sistemas afectados",
            color_discrete_sequence=["#0F2C59", "#1D5B79", "#E59838", "#C0392B", "#95A5A6"],
            template="plotly_white"
        )
        st.plotly_chart(fig_est, use_container_width=True)

with tab3:
    st.subheader("Estimación exploratoria de riesgo de continuidad TIC (Random Forest)")
    st.caption("Nota metodológica: Modelo de clasificación binaria entrenado sobre la base depurada (split 80/20 estratificado). Sus resultados son orientativos y no constituyen certezas operativas.")

    df_ml = df[df["riesgo_continuidad_tic"].isin(["Sí", "No"])].copy()
    feats = ["departamento", "tipo_entidad", "nivel_afectacion", "estado_sistemas"]
    X = df_ml[feats]
    y = (df_ml["riesgo_continuidad_tic"] == "Sí").astype(int)

    if len(df_ml) >= 30 and y.nunique() == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        pipe = Pipeline([
            ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feats)])),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Exactitud en Prueba (Accuracy)", f"{acc:.1%}")
        m2.metric("F1-Score (Clase Riesgo = Sí)", f"{f1:.3f}")
        m3.metric("Tamaño Muestra Entrenamiento", f"{len(X_train)} registros")

        st.markdown("#### Simulador de perfil institucional")
        sc1, sc2 = st.columns(2)
        in_depto = sc1.selectbox("Departamento de la entidad", sorted(df_ml["departamento"].unique()))
        in_tipo = sc1.selectbox("Tipo de entidad", sorted(df_ml["tipo_entidad"].unique()))
        in_nivel = sc2.selectbox("Nivel general de afectación", sorted(df_ml["nivel_afectacion"].unique()))
        in_estado = sc2.selectbox("Estado general de sistemas", sorted(df_ml["estado_sistemas"].unique()))

        muestra_pred = pd.DataFrame([{
            "departamento": in_depto,
            "tipo_entidad": in_tipo,
            "nivel_afectacion": in_nivel,
            "estado_sistemas": in_estado
        }])
        prob_riesgo = pipe.predict_proba(muestra_pred)[0][1]
        st.info(f"**Probabilidad estimada de riesgo en continuidad de servicios esenciales:** `{prob_riesgo:.1%}`")

with tab4:
    cols_mostrar = [c for c in ["departamento", "municipio", "tipo_entidad", "nombre_entidad", "nivel_afectacion", "riesgo_continuidad_tic", "estado_sistemas"] if c in df_filtrado.columns]
    st.dataframe(df_filtrado[cols_mostrar], use_container_width=True, height=420)
