# -*- coding: utf-8 -*-
"""
GeoVisualizador de Vulnerabilidad - Cuenca de Corral
Trabajo Final - Aplicaciones SIG - Gestión Territorial con SIG y TICs
Universidad Austral de Chile, 2026

Temática: vulnerabilidad frente a remoción en masa e inundaciones en la
comuna de Corral (Región de Los Ríos), a partir de:
  - Límite de cuenca hidrográfica costera (IDE Chile)
  - Localidades urbanas/rurales con datos censales (INE)
  - Red hidrográfica (DGA / BCN)
  - Red vial (MOP)
  - Modelo de Elevación Digital 30 m (SRTM)
"""

import base64
import io

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.plugins import Fullscreen, HeatMap, MeasureControl, MiniMap
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from PIL import Image
from shapely.geometry import GeometryCollection, Point
from streamlit_folium import st_folium

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoVisualizador Vulnerabilidad Corral",
    page_icon="🌊",
    layout="wide",
)

DATA_DIR = "data"
CRS_METRICO = "EPSG:32718"  # UTM 18S -> permite calcular áreas/largos en metros
CRS_MAPA = "EPSG:4326"      # WGS84 -> requerido por folium/Leaflet

RUTAS = {
    "cuenca": f"{DATA_DIR}/cuenca_corral.gpkg",
    "poblacion": f"{DATA_DIR}/pobl_urb_corral.gpkg",
    "rios": f"{DATA_DIR}/red_hidrografica.gpkg",
    "vial": f"{DATA_DIR}/red_vial.gpkg",
    "dem": f"{DATA_DIR}/DEM_30m.tif",
}

# ----------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS (con caché para no releer en cada interacción)
# ----------------------------------------------------------------------------

@st.cache_data
def cargar_vectores():
    """Carga las 3 capas vectoriales y las entrega en dos proyecciones:
    una métrica (para calcular áreas/longitudes reales) y una geográfica
    (para dibujar en el mapa con folium)."""
    cuenca = gpd.read_file(RUTAS["cuenca"])
    poblacion = gpd.read_file(RUTAS["poblacion"])
    rios = gpd.read_file(RUTAS["rios"])
    vial = gpd.read_file(RUTAS["vial"])

    # red_vial viene en EPSG:5360, el resto en EPSG:32718: unificamos todo
    capas = {"cuenca": cuenca, "poblacion": poblacion, "rios": rios, "vial": vial}
    salida_metrico, salida_mapa = {}, {}
    for nombre, gdf in capas.items():
        gdf = gdf.to_crs(CRS_METRICO)
        gdf["geometry"] = gdf.geometry.force_2d()  # limpia geometrías 3D/medidas
        # Convierte columnas de fecha a texto (folium/JSON no serializa Timestamp)
        for col in gdf.columns:
            if col != "geometry" and pd.api.types.is_datetime64_any_dtype(gdf[col]):
                gdf[col] = gdf[col].astype(str)
        salida_metrico[nombre] = gdf
        salida_mapa[nombre] = gdf.to_crs(CRS_MAPA)

    # Nombre de localidad legible (ENTIDAD siempre existe; LOCALIDAD a veces es NaN)
    for d in (salida_metrico, salida_mapa):
        d["poblacion"]["NOMBRE_LOC"] = d["poblacion"]["ENTIDAD"].fillna(
            d["poblacion"]["LOCALIDAD"]
        )

    return salida_metrico, salida_mapa


@st.cache_data
def cargar_dem():
    """Lee el DEM y calcula la pendiente (grados) usando una conversión
    aproximada de grados a metros según la latitud media de la escena.
    Es una estimación rápida -no un cálculo geodésico exacto- pero es
    suficiente para clasificar zonas susceptibles a remoción en masa."""
    with rasterio.open(RUTAS["dem"]) as src:
        dem = src.read(1).astype(float)
        nodata = src.nodata
        transform = src.transform
        bounds = src.bounds
    dem[dem == nodata] = np.nan

    lat_media = (bounds.top + bounds.bottom) / 2
    m_por_grado_lat = 111_320
    m_por_grado_lon = 111_320 * np.cos(np.radians(lat_media))
    tam_px_x = transform.a * m_por_grado_lon
    tam_px_y = -transform.e * m_por_grado_lat

    dy, dx = np.gradient(dem, tam_px_y, tam_px_x)
    pendiente = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    return dem, pendiente, bounds, transform


def clasificar_pendiente(pendiente):
    """Clasifica la pendiente en 4 clases de susceptibilidad a remoción en
    masa. Umbrales simplificados con fines didácticos (a mayor pendiente,
    mayor susceptibilidad)."""
    clases = np.full(pendiente.shape, np.nan)
    clases[pendiente < 5] = 0                          # Baja
    clases[(pendiente >= 5) & (pendiente < 15)] = 1    # Moderada
    clases[(pendiente >= 15) & (pendiente < 30)] = 2   # Alta
    clases[pendiente >= 30] = 3                        # Muy alta
    return clases


def array_a_png_overlay(array, cmap, vmin, vmax, bounds):
    """Convierte un array 2D en una imagen PNG RGBA (con transparencia en
    NaN) lista para superponer en folium como ImageOverlay."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(array))
    rgba[np.isnan(array)] = (0, 0, 0, 0)  # transparente donde no hay dato
    img = Image.fromarray((rgba * 255).astype(np.uint8))
    bounds_folium = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return img, bounds_folium


def img_a_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ----------------------------------------------------------------------------
# ÍNDICE DE VULNERABILIDAD HABITACIONAL (por localidad)
# ----------------------------------------------------------------------------

def calcular_indice_vulnerabilidad(gdf_poblacion):
    """Construye un índice simple (0-100) combinando variables censales
    asociadas a vulnerabilidad frente a remoción en masa/inundaciones:
    viviendas irrecuperables, hacinamiento, materialidad precaria y
    abastecimiento de agua desde río/vertiente (mayor exposición a crecidas).
    Cada variable se normaliza como % de los hogares/viviendas de la
    localidad y luego se promedia."""
    df = gdf_poblacion.copy()
    df["pct_irrecuperables"] = 100 * df["n_viv_irrecuperables"] / df["n_vp"].replace(0, np.nan)
    df["pct_hacinadas"] = 100 * df["n_viv_hacinadas"] / df["n_vp"].replace(0, np.nan)
    df["pct_paredes_precarias"] = 100 * df["n_mat_paredes_precarios"] / df["n_vp"].replace(0, np.nan)
    df["pct_agua_rio"] = 100 * df["n_fuente_agua_rio"] / df["n_hog"].replace(0, np.nan)

    componentes = ["pct_irrecuperables", "pct_hacinadas", "pct_paredes_precarias", "pct_agua_rio"]
    df[componentes] = df[componentes].fillna(0)
    df["indice_vulnerabilidad"] = df[componentes].mean(axis=1).round(1)
    return df


# ----------------------------------------------------------------------------
# CARGA DE DATOS
# ----------------------------------------------------------------------------
capas_m, capas_geo = cargar_vectores()
dem, pendiente, dem_bounds, dem_transform = cargar_dem()
clases_pendiente = clasificar_pendiente(pendiente)
capas_geo["poblacion"] = calcular_indice_vulnerabilidad(capas_geo["poblacion"])
capas_m["poblacion"] = calcular_indice_vulnerabilidad(capas_m["poblacion"])

# ----------------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------------
st.sidebar.title("🌊 Capas y controles")
st.sidebar.caption("Vulnerabilidad frente a remoción en masa e inundaciones · Cuenca de Corral")

st.sidebar.markdown("### 🗺️ Mapa base")
mapa_base = st.sidebar.radio(
    "Selecciona el mapa base",
    ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark", "Satélite (Esri)"],
    index=0,
)

st.sidebar.markdown("### 📚 Capas disponibles")
mostrar_cuenca = st.sidebar.checkbox("Límite de cuenca", value=True)
mostrar_poblacion = st.sidebar.checkbox("Localidades (población)", value=True)
mostrar_rios = st.sidebar.checkbox("Red hidrográfica", value=True)
mostrar_vial = st.sidebar.checkbox("Red vial", value=True)
mostrar_dem = st.sidebar.checkbox("DEM (elevación)", value=False)
mostrar_pendiente = st.sidebar.checkbox("Susceptibilidad a remoción en masa (pendiente)", value=True)
mostrar_buffer = st.sidebar.checkbox("Zona de riesgo de inundación (buffer de ríos)", value=True)
mostrar_heatmap = st.sidebar.checkbox("Mapa de calor de población", value=False)

st.sidebar.markdown("### 🎨 Estilo de localidades")
modo_poblacion = st.sidebar.radio(
    "Colorear localidades por:",
    ["Categoría (Aldea/Pueblo)", "Índice de vulnerabilidad"],
    index=1,
)

st.sidebar.markdown("### 🔎 Filtros interactivos")
orden_strahler = st.sidebar.slider(
    "Orden de Strahler mínimo (red hidrográfica)",
    min_value=int(capas_m["rios"]["strahler_n"].min()),
    max_value=int(capas_m["rios"]["strahler_n"].max()),
    value=int(capas_m["rios"]["strahler_n"].min()),
    help="Cauces con mayor orden = mayor caudal y relevancia en la red de drenaje.",
)
clases_camino = sorted(capas_m["vial"]["CLASIFICAC"].dropna().unique())
filtro_vial = st.sidebar.multiselect(
    "Clasificación de caminos a mostrar", clases_camino, default=clases_camino
)

st.sidebar.markdown("### 💧 Análisis espacial")
buffer_m = st.sidebar.slider(
    "Distancia de buffer de inundación (m)", min_value=25, max_value=500, value=100, step=25
)

# ----------------------------------------------------------------------------
# APLICAR FILTROS
# ----------------------------------------------------------------------------
rios_filtrado_geo = capas_geo["rios"][capas_geo["rios"]["strahler_n"] >= orden_strahler]
rios_filtrado_m = capas_m["rios"][capas_m["rios"]["strahler_n"] >= orden_strahler]

vial_filtrado_geo = capas_geo["vial"][capas_geo["vial"]["CLASIFICAC"].isin(filtro_vial)]
vial_filtrado_m = capas_m["vial"][capas_m["vial"]["CLASIFICAC"].isin(filtro_vial)]

# Buffer de riesgo de inundación (zona métrica -> reproyectada para el mapa)
if rios_filtrado_m.empty:
    buffer_geom_m = GeometryCollection()
else:
    buffer_geom_m = rios_filtrado_m.geometry.buffer(buffer_m).union_all()
buffer_gdf_m = gpd.GeoDataFrame(geometry=[buffer_geom_m], crs=CRS_METRICO)
buffer_gdf_geo = buffer_gdf_m.to_crs(CRS_MAPA)

# ----------------------------------------------------------------------------
# ANÁLISIS ESPACIAL: EXPOSICIÓN A INUNDACIÓN (intersección con buffer)
# ----------------------------------------------------------------------------
poblacion_m = capas_m["poblacion"]
poblacion_expuesta = poblacion_m[poblacion_m.intersects(buffer_geom_m)]

vial_m_total = vial_filtrado_m.copy()
vial_m_total["expuesto"] = vial_m_total.intersects(buffer_geom_m)
km_vial_expuesto = vial_m_total.loc[vial_m_total["expuesto"], "geometry"].length.sum() / 1000
km_vial_total = vial_m_total.geometry.length.sum() / 1000

lat_media_dem = (dem_bounds.top + dem_bounds.bottom) / 2
area_px_km2 = (
    abs(dem_transform.a) * 111_320 * np.cos(np.radians(lat_media_dem))
    * abs(dem_transform.e) * 111_320
) / 1e6
area_pendiente_alta_km2 = np.nansum(np.isin(clases_pendiente, [2, 3])) * area_px_km2

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("🌊 GeoVisualizador de Vulnerabilidad — Cuenca de Corral")
st.caption(
    "Análisis de vulnerabilidad frente a remoción en masa e inundaciones · "
    "Comuna de Corral, Región de Los Ríos, Chile"
)

col_mapa, col_panel = st.columns([3, 1])

# ----------------------------------------------------------------------------
# PANEL DE ESTADÍSTICAS
# ----------------------------------------------------------------------------
with col_panel:
    st.markdown("### 📊 Panel de estadísticas")
    st.metric("Población total en localidades", int(poblacion_m["n_per"].sum()))
    st.metric(
        "Población en zona de riesgo de inundación",
        int(poblacion_expuesta["n_per"].sum()),
        help="Localidades cuyo polígono intersecta el buffer de ríos definido en el panel lateral.",
    )
    st.metric(
        "Red vial expuesta a inundación",
        f"{km_vial_expuesto:.1f} / {km_vial_total:.1f} km",
    )
    st.metric(
        "Área con pendiente alta/muy alta (>15°)",
        f"{area_pendiente_alta_km2:.1f} km²",
        help="Zonas con mayor susceptibilidad a remoción en masa dentro del área cubierta por el DEM.",
    )

    st.markdown("#### Vulnerabilidad por localidad")
    fig, ax = plt.subplots(figsize=(4, 2.6))
    datos_barra = capas_m["poblacion"].sort_values("indice_vulnerabilidad")
    colores_barra = plt.cm.YlOrRd(Normalize(0, 100)(datos_barra["indice_vulnerabilidad"]))
    ax.barh(datos_barra["NOMBRE_LOC"], datos_barra["indice_vulnerabilidad"], color=colores_barra)
    ax.set_xlabel("Índice de vulnerabilidad (0-100)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# CONSTRUCCIÓN DEL MAPA
# ----------------------------------------------------------------------------
TILES = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Positron": "CartoDB positron",
    "CartoDB Dark": "CartoDB dark_matter",
}

with col_mapa:
    centro = [
        (capas_geo["cuenca"].total_bounds[1] + capas_geo["cuenca"].total_bounds[3]) / 2,
        (capas_geo["cuenca"].total_bounds[0] + capas_geo["cuenca"].total_bounds[2]) / 2,
    ]

    if mapa_base == "Satélite (Esri)":
        m = folium.Map(location=centro, zoom_start=11, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satélite (Esri)",
        ).add_to(m)
    else:
        m = folium.Map(location=centro, zoom_start=11, tiles=TILES[mapa_base])

    # --- Capa: límite de cuenca (solo borde) ---
    if mostrar_cuenca:
        folium.GeoJson(
            capas_geo["cuenca"],
            name="Límite de cuenca",
            style_function=lambda x: {
                "fillOpacity": 0,
                "color": "#2c3e50",
                "weight": 3,
                "dashArray": "6,4",
            },
            tooltip=folium.GeoJsonTooltip(fields=["NOM_CUEN", "COD_CUEN"],
                                           aliases=["Cuenca:", "Código:"]),
        ).add_to(m)

    # --- Capa: raster de pendiente / susceptibilidad a remoción en masa ---
    if mostrar_pendiente:
        cmap_pend = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"])
        img, b = array_a_png_overlay(clases_pendiente, cmap_pend, 0, 3, dem_bounds)
        folium.raster_layers.ImageOverlay(
            image=img_a_data_url(img),
            bounds=b,
            opacity=0.55,
            name="Susceptibilidad a remoción en masa",
        ).add_to(m)

    # --- Capa: DEM elevación ---
    if mostrar_dem:
        img_dem, b_dem = array_a_png_overlay(dem, plt.cm.terrain, np.nanmin(dem), np.nanmax(dem), dem_bounds)
        folium.raster_layers.ImageOverlay(
            image=img_a_data_url(img_dem),
            bounds=b_dem,
            opacity=0.6,
            name="DEM (elevación)",
        ).add_to(m)

    # --- Capa: buffer de riesgo de inundación ---
    if mostrar_buffer:
        folium.GeoJson(
            buffer_gdf_geo,
            name=f"Zona de riesgo de inundación ({buffer_m} m)",
            style_function=lambda x: {
                "fillColor": "#3498db",
                "color": "#2980b9",
                "weight": 1,
                "fillOpacity": 0.25,
            },
        ).add_to(m)

    # --- Capa: red hidrográfica ---
    if mostrar_rios and not rios_filtrado_geo.empty:
        def estilo_rio(feature):
            orden = feature["properties"].get("strahler_n") or 1
            return {
                "color": "#1f77b4",
                "weight": 1 + orden,
                "opacity": 0.85,
            }

        folium.GeoJson(
            rios_filtrado_geo,
            name="Red hidrográfica",
            style_function=estilo_rio,
            tooltip=folium.GeoJsonTooltip(
                fields=["nom_ssubc", "tipo_bcn", "strahler_n"],
                aliases=["Subcuenca:", "Tipo:", "Orden Strahler:"],
            ),
        ).add_to(m)
    elif mostrar_rios:
        st.info("No hay cauces con ese orden de Strahler mínimo.")

    # --- Capa: red vial ---
    if mostrar_vial and not vial_filtrado_geo.empty:
        colores_vial = {
            "Camino Regional Comunal": "#8e44ad",
            "Camino Regional Provincial": "#c0392b",
            "Camino Regional de Acceso": "#7f8c8d",
        }

        def estilo_vial(feature):
            clas = feature["properties"].get("CLASIFICAC")
            return {"color": colores_vial.get(clas, "#34495e"), "weight": 3, "opacity": 0.9}

        folium.GeoJson(
            vial_filtrado_geo,
            name="Red vial",
            style_function=estilo_vial,
            tooltip=folium.GeoJsonTooltip(
                fields=["NOMBRE_CAM", "CLASIFICAC", "CARPETA"],
                aliases=["Camino:", "Clasificación:", "Carpeta:"],
            ),
        ).add_to(m)
    elif mostrar_vial:
        st.info("No hay caminos que coincidan con la clasificación seleccionada.")

    # --- Capa: localidades / población ---
    if mostrar_poblacion:
        poblacion_geo = capas_geo["poblacion"]
        if modo_poblacion == "Categoría (Aldea/Pueblo)":
            colores_cat = {"Aldea": "#27ae60", "Pueblo": "#e67e22"}

            def estilo_pob(feature):
                cat = feature["properties"].get("CATEGORIA")
                return {"fillColor": colores_cat.get(cat, "#95a5a6"), "color": "#2c3e50",
                        "weight": 1.5, "fillOpacity": 0.55}
        else:
            cmap_vuln = plt.cm.YlOrRd
            norm_vuln = Normalize(0, 100)

            def estilo_pob(feature):
                val = feature["properties"].get("indice_vulnerabilidad") or 0
                r, g, b_, a = cmap_vuln(norm_vuln(val))
                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b_*255):02x}"
                return {"fillColor": color, "color": "#2c3e50", "weight": 1.5, "fillOpacity": 0.65}

        folium.GeoJson(
            poblacion_geo,
            name="Localidades (población)",
            style_function=estilo_pob,
            tooltip=folium.GeoJsonTooltip(
                fields=["NOMBRE_LOC", "CATEGORIA", "n_per", "n_vp", "indice_vulnerabilidad"],
                aliases=["Localidad:", "Categoría:", "Población:", "Viviendas:", "Índice vulnerab.:"],
            ),
        ).add_to(m)

        if mostrar_heatmap:
            centroides_m = capas_m["poblacion"].geometry.centroid
            centroides = gpd.GeoSeries(centroides_m, crs=CRS_METRICO).to_crs(CRS_MAPA)
            puntos_calor = [
                [pt.y, pt.x, max(row["n_per"], 1)]
                for pt, (_, row) in zip(centroides, poblacion_geo.iterrows())
            ]
            HeatMap(puntos_calor, name="Densidad de población", radius=35, blur=25).add_to(m)

    # --- Plugins ---
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(primary_length_unit="meters").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, use_container_width=True, height=600, returned_objects=[])

    if mostrar_pendiente:
        st.markdown(
            "**Leyenda — Susceptibilidad a remoción en masa (pendiente):** "
            "🟩 Baja (<5°)  🟨 Moderada (5°-15°)  🟧 Alta (15°-30°)  🟥 Muy alta (>30°)"
        )

# ----------------------------------------------------------------------------
# PERFIL DE ELEVACIÓN
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### ⛰️ Perfil de elevación")
st.caption(
    "Muestra el perfil altitudinal entre dos puntos sobre el DEM, útil para "
    "visualizar quiebres de pendiente asociados a procesos de remoción en masa."
)

bounds_geo = capas_geo["cuenca"].total_bounds  # minx, miny, maxx, maxy
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Punto A**")
    lat_a = st.number_input("Latitud A", value=float(bounds_geo[3]) - 0.02, format="%.5f")
    lon_a = st.number_input("Longitud A", value=float((bounds_geo[0] + bounds_geo[2]) / 2), format="%.5f")
with col2:
    st.markdown("**Punto B**")
    lat_b = st.number_input("Latitud B", value=float(bounds_geo[1]) + 0.02, format="%.5f")
    lon_b = st.number_input("Longitud B", value=float((bounds_geo[0] + bounds_geo[2]) / 2), format="%.5f")

n_muestras = 200
lats = np.linspace(lat_a, lat_b, n_muestras)
lons = np.linspace(lon_a, lon_b, n_muestras)

with rasterio.open(RUTAS["dem"]) as src:
    valores = list(src.sample(zip(lons, lats)))
    nodata_dem = src.nodata
elevaciones = [v[0] if v[0] != nodata_dem else np.nan for v in valores]

punto_a = gpd.GeoSeries([Point(lon_a, lat_a)], crs=CRS_MAPA).to_crs(CRS_METRICO)
punto_b = gpd.GeoSeries([Point(lon_b, lat_b)], crs=CRS_MAPA).to_crs(CRS_METRICO)
distancia_total_km = punto_a.distance(punto_b).iloc[0] / 1000
eje_x = np.linspace(0, distancia_total_km, n_muestras)

fig2, ax2 = plt.subplots(figsize=(9, 3))
ax2.fill_between(eje_x, elevaciones, color="#95a5a6", alpha=0.4)
ax2.plot(eje_x, elevaciones, color="#2c3e50", linewidth=1.5)
ax2.set_xlabel("Distancia (km)")
ax2.set_ylabel("Elevación (m s.n.m.)")
ax2.set_title("Perfil de elevación A → B")
ax2.grid(alpha=0.3)
st.pyplot(fig2)
plt.close(fig2)

# ----------------------------------------------------------------------------
# TABLA DE ATRIBUTOS
# ----------------------------------------------------------------------------
with st.expander("📋 Tabla de atributos — Localidades"):
    columnas_tabla = [
        "NOMBRE_LOC", "CATEGORIA", "n_per", "n_hog", "n_vp",
        "n_viv_irrecuperables", "n_viv_hacinadas", "n_mat_paredes_precarios",
        "n_fuente_agua_rio", "indice_vulnerabilidad",
    ]
    st.dataframe(capas_m["poblacion"][columnas_tabla], width="stretch")

st.caption(
    "Fuentes: cuencas hidrográficas y red hidrográfica (IDE Chile / BCN), "
    "red vial (MOP), datos censales (INE), DEM 30 m (SRTM). Índice de "
    "vulnerabilidad construido con fines académicos; no constituye un "
    "instrumento oficial de gestión de riesgo."
)
