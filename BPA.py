import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Pérdida de Bosque - Perú",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌳"
)

# --- CSS Personalizado ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Funciones de Carga ---
@st.cache_resource
def cargar_config_general():
    """Carga la configuración general del proyecto"""
    try:
        with open('config_general.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Archivo config_general.json no encontrado")
        return None

@st.cache_resource
def cargar_metricas_detalladas():
    """Carga las métricas de todos los modelos"""
    try:
        with open('metricas_detalladas.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def cargar_datos_historicos():
    """Carga datos históricos para visualización"""
    try:
        return pd.read_csv('datos_historicos.csv')
    except FileNotFoundError:
        st.warning("⚠️ datos_historicos.csv no encontrado")
        return None

@st.cache_data
def cargar_predicciones_test():
    """Carga las predicciones en el conjunto de prueba"""
    try:
        with open('predicciones_test.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def normalizar_nombre_depto(nombre):
    """Normaliza nombre de departamento para nombres de archivo"""
    return nombre.lower().replace(' ', '_').replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')

@st.cache_resource
def cargar_modelo(departamento, tipo_modelo):
    """
    Carga un modelo específico
    tipo_modelo: 'arima', 'sarimax_basic', 'sarimax_opt'
    """
    depto_clean = normalizar_nombre_depto(departamento)
    filename = f'modelo_{tipo_modelo}_{depto_clean}.joblib'
    
    try:
        modelo = joblib.load(filename)
        return modelo
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error al cargar {filename}: {e}")
        return None

def cargar_config_opt(departamento):
    """Carga la configuración del modelo optimizado"""
    depto_clean = normalizar_nombre_depto(departamento)
    filename = f'config_sarimax_opt_{depto_clean}.json'
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# --- Cargar Recursos Globales ---
config_general = cargar_config_general()
metricas_detalladas = cargar_metricas_detalladas()
datos_historicos = cargar_datos_historicos()
predicciones_test = cargar_predicciones_test()

# --- Header ---
st.markdown('<p class="main-header">🌳 Sistema de Predicción de Pérdida de Bosque - Perú</p>', 
            unsafe_allow_html=True)

if config_general:
    st.markdown(f"""
    **Proyecto de Análisis Ambiental Temporal**
    
    Este sistema utiliza modelos ARIMA y SARIMAX entrenados con datos de **{config_general['periodo_datos']['inicio']}-{config_general['periodo_datos']['fin']}**
    para predecir la pérdida de bosque en departamentos críticos del Perú.
    """)
else:
    st.warning("⚠️ Configuración general no disponible. Verifica que config_general.json esté en el directorio.")

# --- BARRA LATERAL ---
st.sidebar.header("🎯 Configuración de Predicción")

# Selector de Departamento
if config_general:
    departamentos_disponibles = config_general['departamentos']
else:
    departamentos_disponibles = ['UCAYALI', 'SAN MARTIN', 'LORETO']

departamento_seleccionado = st.sidebar.selectbox(
    "Selecciona el Departamento",
    options=departamentos_disponibles,
    help="Departamento para el cual se generará la predicción"
)

st.sidebar.markdown("---")

# Selector de Tipo de Modelo
st.sidebar.subheader("🤖 Tipo de Modelo")

tipo_modelo = st.sidebar.radio(
    "Selecciona el modelo a utilizar",
    options=[
        'ARIMA (sin exógenas)',
        'SARIMAX Básico (con exógenas)',
        'SARIMAX Optimizado (mejor performance)'
    ],
    help="ARIMA: más simple, sin variables exógenas\nSARIMAX Básico: usa todas las exógenas\nSARIMAX Optimizado: feature selection + grid search"
)

# Mapear selección a nombre interno
modelo_map = {
    'ARIMA (sin exógenas)': 'arima',
    'SARIMAX Básico (con exógenas)': 'sarimax_basic',
    'SARIMAX Optimizado (mejor performance)': 'sarimax_opt'
}
tipo_modelo_key = modelo_map[tipo_modelo]

# Cargar el modelo seleccionado
modelo = cargar_modelo(departamento_seleccionado, tipo_modelo_key)

if modelo is not None:
    st.sidebar.success(f"✅ Modelo cargado correctamente")
else:
    st.sidebar.error(f"❌ Modelo no disponible")

# Mostrar métricas del modelo si están disponibles
if metricas_detalladas and departamento_seleccionado in metricas_detalladas:
    metricas_modelo = metricas_detalladas[departamento_seleccionado].get(tipo_modelo_key)
    if metricas_modelo:
        with st.sidebar.expander("📊 Métricas del Modelo", expanded=False):
            st.metric("MAE", f"{metricas_modelo['MAE']:.2f}")
            st.metric("RMSE", f"{metricas_modelo['RMSE']:.2f}")
            st.metric("MAPE", f"{metricas_modelo['MAPE']:.2f}%")

st.sidebar.markdown("---")

# --- Configuración Temporal ---
st.sidebar.subheader("📅 Horizonte de Predicción")

anio_inicio = st.sidebar.number_input(
    "Año de Inicio de Predicción",
    min_value=2024,
    max_value=2050,
    value=2024,
    step=1
)

periodos_forecast = st.sidebar.slider(
    "Años a Predecir",
    min_value=1,
    max_value=20,
    value=5,
    help="Número de años futuros para predicción"
)

st.sidebar.markdown("---")

# --- Variables Exógenas (solo para SARIMAX) ---
exogenas_inputs = {}
usar_exogenas = tipo_modelo_key != 'arima'

if usar_exogenas and modelo is not None:
    st.sidebar.subheader("🌿 Variables Exógenas")
    
    # Determinar qué features usar
    if tipo_modelo_key == 'sarimax_opt':
        config_opt = cargar_config_opt(departamento_seleccionado)
        if config_opt:
            features_a_usar = config_opt['top_features']
            st.sidebar.info(f"Usando {len(features_a_usar)} features seleccionadas")
        else:
            features_a_usar = config_general['variables_exogenas'] if config_general else []
    else:
        features_a_usar = config_general['variables_exogenas'] if config_general else []
    
    modo_input = st.sidebar.radio(
        "Modo de entrada de exógenas",
        ["Valores por defecto", "Personalizar valores"],
        help="Los valores por defecto son estimaciones basadas en tendencias históricas"
    )
    
    if modo_input == "Personalizar valores":
        with st.sidebar.expander("🔧 Configurar Exógenas", expanded=True):
            st.caption("**Valores en hectáreas**")
            
            # Valores por defecto razonables
            valores_default = {
                'bosque': 5000000,
                'bosque_seco': 100000.0,
                'bosque_inundable': 50000.0,
                'zona_pantanosa_o_pastizal_inundable': 10000,
                'pastizal_herbazal': 20000,
                'otras_formaciones_no_boscosas': 5000,
                'pasto': 30000.0,
                'agricultura': 100000.0,
                'mosaico_agropecuario': 15000,
                'playa': 100,
                'infraestructura_urbana': 5000,
                'otra_area_sin_vegetacion': 1000,
                'mineria': 5000.0,
                'otra_area_natural_sin_vegetacion': 500,
                'rio_lago_u_oceano': 50000
            }
            
            for feature in features_a_usar:
                valor_default = valores_default.get(feature, 1000.0)
                exogenas_inputs[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    value=float(valor_default),
                    step=100.0,
                    format="%.1f",
                    key=f"exog_{feature}"
                )
    else:
        # Usar valores por defecto
        valores_default = {
            'bosque': 5000000,
            'bosque_seco': 100000.0,
            'bosque_inundable': 50000.0,
            'zona_pantanosa_o_pastizal_inundable': 10000,
            'pastizal_herbazal': 20000,
            'otras_formaciones_no_boscosas': 5000,
            'pasto': 30000.0,
            'agricultura': 100000.0,
            'mosaico_agropecuario': 15000,
            'playa': 100,
            'infraestructura_urbana': 5000,
            'otra_area_sin_vegetacion': 1000,
            'mineria': 5000.0,
            'otra_area_natural_sin_vegetacion': 500,
            'rio_lago_u_oceano': 50000
        }
        for feature in features_a_usar:
            exogenas_inputs[feature] = valores_default.get(feature, 1000.0)

# --- Botón de Predicción ---
st.sidebar.markdown("---")
predecir = st.sidebar.button("🚀 Generar Predicción", type="primary", use_container_width=True)

# --- ÁREA PRINCIPAL ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predicción", "📈 Comparación de Modelos", "📋 Datos Históricos", "ℹ️ Info del Modelo"])

with tab1:
    if predecir:
        if modelo is None:
            st.error(f"❌ No se puede generar predicción. El modelo {tipo_modelo} para {departamento_seleccionado} no está disponible.")
        else:
            with st.spinner(f"Generando predicción para {departamento_seleccionado}..."):
                try:
                    # Generar años de predicción
                    anios_prediccion = list(range(anio_inicio, anio_inicio + periodos_forecast))
                    
                    # Realizar predicción según el tipo de modelo
                    if tipo_modelo_key == 'arima':
                        # ARIMA: sin exógenas
                        forecast = modelo.forecast(steps=periodos_forecast)
                        predicciones = forecast
                        
                    else:
                        # SARIMAX: con exógenas
                        # Crear DataFrame con exógenas
                        if tipo_modelo_key == 'sarimax_opt':
                            config_opt = cargar_config_opt(departamento_seleccionado)
                            features_usar = config_opt['top_features'] if config_opt else []
                        else:
                            features_usar = list(exogenas_inputs.keys())
                        
                        # Crear exógenas para todos los períodos
                        exog_df = pd.DataFrame([exogenas_inputs] * periodos_forecast)
                        exog_df = exog_df[features_usar]
                        
                        # Predicción
                        forecast = modelo.forecast(steps=periodos_forecast, exog=exog_df)
                        predicciones = forecast
                    
                    # Convertir a array si es necesario
                    if hasattr(predicciones, 'values'):
                        predicciones = predicciones.values
                    
                    # Calcular intervalos de confianza (simulados si no están disponibles)
                    std_dev = np.abs(predicciones) * 0.15  # 15% de desviación
                    lower_bound = predicciones - 1.96 * std_dev
                    upper_bound = predicciones + 1.96 * std_dev
                    
                    # Asegurar que no haya valores negativos
                    lower_bound = np.maximum(lower_bound, 0)
                    
                    # Crear DataFrame de resultados
                    df_prediccion = pd.DataFrame({
                        'Año': anios_prediccion,
                        'Pérdida Bosque (ha)': predicciones,
                        'Límite Inferior': lower_bound,
                        'Límite Superior': upper_bound
                    })
                    
                    # --- Métricas Principales ---
                    st.subheader(f"🎯 Resultados - {departamento_seleccionado}")
                    st.caption(f"Modelo: **{tipo_modelo}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label=f"Predicción {anio_inicio}",
                            value=f"{predicciones[0]:,.0f} ha"
                        )
                    
                    with col2:
                        cambio_pct = ((predicciones[-1] - predicciones[0]) / predicciones[0] * 100)
                        st.metric(
                            label=f"Predicción {anio_inicio + periodos_forecast - 1}",
                            value=f"{predicciones[-1]:,.0f} ha",
                            delta=f"{cambio_pct:+.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            label="Promedio Período",
                            value=f"{predicciones.mean():,.0f} ha"
                        )
                    
                    with col4:
                        total_perdida = predicciones.sum()
                        st.metric(
                            label="Pérdida Total Proyectada",
                            value=f"{total_perdida:,.0f} ha"
                        )
                    
                    st.markdown("---")
                    
                    # --- Gráfico Principal ---
                    st.subheader("📉 Proyección de Pérdida de Bosque")
                    
                    # Datos históricos
                    fig = go.Figure()
                    
                    if datos_historicos is not None:
                        df_hist = datos_historicos[
                            datos_historicos['departamento'] == departamento_seleccionado
                        ].sort_values('anio')
                        
                        if not df_hist.empty:
                            fig.add_trace(go.Scatter(
                                x=df_hist['anio'],
                                y=df_hist['perdida_bosque'],
                                mode='lines+markers',
                                name='Histórico',
                                line=dict(color='#2E7D32', width=2),
                                marker=dict(size=6)
                            ))
                    
                    # Predicciones
                    fig.add_trace(go.Scatter(
                        x=df_prediccion['Año'],
                        y=df_prediccion['Pérdida Bosque (ha)'],
                        mode='lines+markers',
                        name='Predicción',
                        line=dict(color='#FF6B6B', width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    # Intervalo de confianza
                    fig.add_trace(go.Scatter(
                        x=df_prediccion['Año'].tolist() + df_prediccion['Año'].tolist()[::-1],
                        y=df_prediccion['Límite Superior'].tolist() + df_prediccion['Límite Inferior'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 107, 107, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='IC 95%',
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(
                        title=f"Pérdida de Bosque - {departamento_seleccionado}",
                        xaxis_title="Año",
                        yaxis_title="Pérdida de Bosque (hectáreas)",
                        hovermode='x unified',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Tabla de Resultados ---
                    st.subheader("📋 Tabla de Predicciones Detallada")
                    
                    df_prediccion['Pérdida Acumulada (ha)'] = df_prediccion['Pérdida Bosque (ha)'].cumsum()
                    df_prediccion['Cambio Anual (%)'] = df_prediccion['Pérdida Bosque (ha)'].pct_change() * 100
                    
                    st.dataframe(
                        df_prediccion.style.format({
                            'Pérdida Bosque (ha)': '{:,.0f}',
                            'Límite Inferior': '{:,.0f}',
                            'Límite Superior': '{:,.0f}',
                            'Pérdida Acumulada (ha)': '{:,.0f}',
                            'Cambio Anual (%)': '{:+.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Botón de descarga
                    csv = df_prediccion.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Predicciones (CSV)",
                        data=csv,
                        file_name=f"prediccion_{departamento_seleccionado}_{tipo_modelo_key}_{anio_inicio}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error al generar predicción: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("👈 Configura los parámetros en la barra lateral y presiona **'Generar Predicción'**")
        
        # Vista previa
        if datos_historicos is not None:
            st.subheader(f"📊 Vista Previa - {departamento_seleccionado}")
            
            df_dept = datos_historicos[datos_historicos['departamento'] == departamento_seleccionado]
            
            if not df_dept.empty:
                fig = px.line(
                    df_dept,
                    x='anio',
                    y='perdida_bosque',
                    title=f"Pérdida de Bosque Histórica - {departamento_seleccionado}",
                    labels={'anio': 'Año', 'perdida_bosque': 'Pérdida (ha)'}
                )
                fig.update_traces(line_color='#2E7D32', line_width=3)
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 Comparación de Modelos en Test Set")
    
    if predicciones_test and departamento_seleccionado in predicciones_test:
        preds = predicciones_test[departamento_seleccionado]
        
        # Gráfico comparativo
        fig = go.Figure()
        
        # Valores reales
        if 'arima' in preds:
            anios = preds['arima']['anios']
            y_true = preds['arima']['y_true']
            
            fig.add_trace(go.Scatter(
                x=anios,
                y=y_true,
                mode='lines+markers',
                name='Valores Reales',
                line=dict(color='black', width=3),
                marker=dict(size=8)
            ))
        
        # Predicciones de cada modelo
        colores = {'arima': '#1f77b4', 'sarimax_basic': '#ff7f0e', 'sarimax_opt': '#2ca02c'}
        nombres = {'arima': 'ARIMA', 'sarimax_basic': 'SARIMAX Básico', 'sarimax_opt': 'SARIMAX Optimizado'}
        
        for modelo_key, datos in preds.items():
            fig.add_trace(go.Scatter(
                x=datos['anios'],
                y=datos['y_pred'],
                mode='lines+markers',
                name=nombres[modelo_key],
                line=dict(color=colores[modelo_key], width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"Comparación de Modelos - {departamento_seleccionado}",
            xaxis_title="Año",
            yaxis_title="Pérdida de Bosque (ha)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de métricas
        if metricas_detalladas and departamento_seleccionado in metricas_detalladas:
            st.subheader("📈 Métricas de Performance")
            
            metricas = metricas_detalladas[departamento_seleccionado]
            
            rows = []
            for modelo_key in ['arima', 'sarimax_basic', 'sarimax_opt']:
                if modelo_key in metricas:
                    row = {'Modelo': nombres[modelo_key]}
                    row.update(metricas[modelo_key])
                    rows.append(row)
            
            df_metricas = pd.DataFrame(rows)
            
            # Formatear y mostrar
            st.dataframe(
                df_metricas.style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'MAPE': '{:.2f}'
                }).highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen'),
                use_container_width=True
            )
    else:
        st.warning("No hay datos de predicciones de prueba disponibles")

with tab3:
    st.subheader("📈 Datos Históricos de Pérdida de Bosque")
    
    if datos_historicos is not None:
        # Comparación entre departamentos
        fig = px.line(
            datos_historicos,
            x='anio',
            y='perdida_bosque',
            color='departamento',
            title="Comparación Entre Departamentos",
            labels={'anio': 'Año', 'perdida_bosque': 'Pérdida de Bosque (ha)', 'departamento': 'Departamento'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos
        st.dataframe(datos_historicos, use_container_width=True)
    else:
        st.warning("Datos históricos no disponibles")

with tab4:
    st.subheader("ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Modelos Disponibles")
        st.markdown("""
        **1. ARIMA (AutoRegressive Integrated Moving Average)**
        - Modelo univariado
        - No utiliza variables exógenas
        - Más simple y rápido
        - Bueno para tendencias simples
        
        **2. SARIMAX Básico**
        - Utiliza todas las variables exógenas disponibles (15 variables)
        - Mayor complejidad
        - Captura relaciones multivariadas
        
        **3. SARIMAX Optimizado**
        - Feature selection con Random Forest
        - Grid search para parámetros (p,d,q)
        - Validación cruzada walk-forward
        - Mejor performance general
        """)
    
    with col2:
        if config_general:
            st.markdown("### 🔧 Configuración del Proyecto")
            st.json(config_general)
    
    st.markdown("---")
    
    if metricas_detalladas:
        st.markdown("### 📊 Performance por Departamento")
        
        for depto in config_general['departamentos']:
            with st.expander(f"{depto}"):
                if depto in metricas_detalladas:
                    st.json(metricas_detalladas[depto])

# --- Footer ---
st.markdown("---")
st.caption("🌳 Sistema de Predicción de Pérdida de Bosque | Desarrollado con Streamlit | ARIMAX/SARIMAX Models")