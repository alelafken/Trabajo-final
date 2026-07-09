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
from rasterio.features import shapes
from shapely.geometry import GeometryCollection, Point, shape
from streamlit_folium import st_folium

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoVisualizador Vulnerabilidad Corral",
    layout="wide",
)

DATA_DIR = "data"
CRS_METRICO = "EPSG:32718"  
CRS_MAPA = "EPSG:4326"      

RUTAS = {
    "cuenca": f"{DATA_DIR}/cuenca_corral.gpkg",
    "poblacion": f"{DATA_DIR}/pobl_urb_corral.gpkg",
    "rios": f"{DATA_DIR}/red_hidrografica.gpkg",
    "vial": f"{DATA_DIR}/red_vial.gpkg",
    "dem": f"{DATA_DIR}/DEM_30m.tif",
}

if 'puntos_perfil' not in st.session_state:
    st.session_state.puntos_perfil = []

# ----------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS 
# ----------------------------------------------------------------------------

@st.cache_data
def cargar_vectores():
    cuenca = gpd.read_file(RUTAS["cuenca"])
    poblacion = gpd.read_file(RUTAS["poblacion"])
    rios = gpd.read_file(RUTAS["rios"])
    vial = gpd.read_file(RUTAS["vial"])

    capas = {"cuenca": cuenca, "poblacion": poblacion, "rios": rios, "vial": vial}
    salida_metrico, salida_mapa = {}, {}
    
    for nombre, gdf in capas.items():
        gdf = gdf.to_crs(CRS_METRICO)
        gdf["geometry"] = gdf.geometry.force_2d()
        for col in gdf.columns:
            if col != "geometry" and pd.api.types.is_datetime64_any_dtype(gdf[col]):
                gdf[col] = gdf[col].astype(str)
        salida_metrico[nombre] = gdf
        salida_mapa[nombre] = gdf.to_crs(CRS_MAPA)

    for d in (salida_metrico, salida_mapa):
        d["poblacion"]["NOMBRE_LOC"] = d["poblacion"]["ENTIDAD"].fillna(d["poblacion"]["LOCALIDAD"])

    return salida_metrico, salida_mapa

@st.cache_data
def cargar_dem():
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
    clases = np.full(pendiente.shape, np.nan)
    clases[pendiente < 5] = 0                          
    clases[(pendiente >= 5) & (pendiente < 15)] = 1    
    clases[(pendiente >= 15) & (pendiente < 30)] = 2   
    clases[pendiente >= 30] = 3                        
    return clases

@st.cache_data
def generar_geometrias_riesgo(_clases_pendiente, _transform):
    # Polígonos de zonas planas (<5 grados)
    mask_plano = (_clases_pendiente == 0).astype('uint8')
    geoms_plano = [shape(g) for g, v in shapes(mask_plano, transform=_transform) if v == 1]
    
    # Polígonos de remoción alta y muy alta (>15 grados)
    mask_rem = np.isin(_clases_pendiente, [2, 3]).astype('uint8')
    geoms_rem = [shape(g) for g, v in shapes(mask_rem, transform=_transform) if v == 1]
    
    plano_gdf = gpd.GeoDataFrame(geometry=geoms_plano, crs=CRS_MAPA) if geoms_plano else gpd.GeoDataFrame(geometry=[], crs=CRS_MAPA)
    rem_gdf = gpd.GeoDataFrame(geometry=geoms_rem, crs=CRS_MAPA) if geoms_rem else gpd.GeoDataFrame(geometry=[], crs=CRS_MAPA)
    
    plano_geom_m = plano_gdf.to_crs(CRS_METRICO).union_all() if not plano_gdf.empty else GeometryCollection()
    rem_geom_m = rem_gdf.to_crs(CRS_METRICO).union_all() if not rem_gdf.empty else GeometryCollection()
    
    return plano_geom_m, rem_geom_m

def calcular_indice_vulnerabilidad_espacial(gdf_poblacion_m, rem_geom_m, buffer_geom_m):
    df = gdf_poblacion_m.copy()
    indices = []
    
    for _, row in df.iterrows():
        area_tot = row.geometry.area
        if area_tot == 0:
            indices.append(0)
            continue
            
        a_rem = row.geometry.intersection(rem_geom_m).area
        a_inu = row.geometry.intersection(buffer_geom_m).area
        pct_afectado = ((a_rem + a_inu) / area_tot) * 100
        indices.append(pct_afectado)
        
    df["indice_vulnerabilidad"] = indices
    max_v = df["indice_vulnerabilidad"].max()
    
    if max_v > 0:
        df["indice_vulnerabilidad"] = (df["indice_vulnerabilidad"] / max_v) * 100
        
    df["indice_vulnerabilidad"] = df["indice_vulnerabilidad"].round(1)
    return df

def array_a_png_overlay(array, cmap, vmin, vmax, bounds):
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(array))
    rgba[np.isnan(array)] = (0, 0, 0, 0)  
    img = Image.fromarray((rgba * 255).astype(np.uint8))
    bounds_folium = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return img, bounds_folium

def img_a_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ----------------------------------------------------------------------------
# CARGA DE DATOS Y GEOMETRÍAS
# ----------------------------------------------------------------------------
capas_m, capas_geo = cargar_vectores()
dem, pendiente, dem_bounds, dem_transform = cargar_dem()
clases_pendiente = clasificar_pendiente(pendiente)

geom_plano_m, geom_remocion_m = generar_geometrias_riesgo(clases_pendiente, dem_transform)

# ----------------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------------
st.sidebar.title("Capas y controles")
st.sidebar.caption("Vulnerabilidad espacial territorial · Cuenca de Corral")

st.sidebar.markdown("### 🗺️ Mapa base")
mapa_base = st.sidebar.radio(
    "Selecciona el mapa base",
    ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark", "Satélite (Esri)"],
    index=0,
)

st.sidebar.markdown("### 📚 Capas disponibles")
mostrar_cuenca = st.sidebar.checkbox("Límite de cuenca", value=True)
mostrar_dem = st.sidebar.checkbox("DEM (elevación)", value=False)
mostrar_pendiente = st.sidebar.checkbox("Susceptibilidad a remoción en masa", value=True)
mostrar_buffer = st.sidebar.checkbox("Zona de riesgo de inundación", value=True)
mostrar_rios = st.sidebar.checkbox("Red hidrográfica", value=True)
mostrar_vial = st.sidebar.checkbox("Red vial", value=True)
mostrar_poblacion = st.sidebar.checkbox("Localidades (población)", value=True)

st.sidebar.markdown("### 🎨 Estilo de localidades")
modo_poblacion = st.sidebar.radio(
    "Colorear localidades por:",
    ["Categoría (Aldea/Pueblo)", "Índice de vulnerabilidad espacial"],
    index=1,
)

st.sidebar.markdown("### 🔎 Parámetros Espaciales")
orden_strahler = st.sidebar.slider("Orden de Strahler mínimo", 1, 6, 1)
buffer_m = st.sidebar.slider("Distancia de buffer de inundación (m)", 25, 500, 100, 25)

clases_camino = sorted(capas_m["vial"]["CLASIFICAC"].dropna().unique())
filtro_vial = st.sidebar.multiselect("Tipos de vía", clases_camino, default=clases_camino)

# ----------------------------------------------------------------------------
# PROCESAMIENTO ESPACIAL DINÁMICO
# ----------------------------------------------------------------------------
rios_filtrado_geo = capas_geo["rios"][capas_geo["rios"]["strahler_n"] >= orden_strahler]
rios_filtrado_m = capas_m["rios"][capas_m["rios"]["strahler_n"] >= orden_strahler]

vial_filtrado_geo = capas_geo["vial"][capas_geo["vial"]["CLASIFICAC"].isin(filtro_vial)]
vial_filtrado_m = capas_m["vial"][capas_m["vial"]["CLASIFICAC"].isin(filtro_vial)]

if rios_filtrado_m.empty:
    buffer_geom_m = GeometryCollection()
else:
    buffer_geom_m = rios_filtrado_m.geometry.buffer(buffer_m).union_all()

capas_m["poblacion"] = calcular_indice_vulnerabilidad_espacial(capas_m["poblacion"], geom_remocion_m, buffer_geom_m)
capas_geo["poblacion"] = capas_m["poblacion"].to_crs(CRS_MAPA)

# Segmentar la inundación según pendiente
buffer_inundacion_alta_m = buffer_geom_m.intersection(geom_plano_m)
buffer_inundacion_mod_m = buffer_geom_m.difference(geom_plano_m)
buffer_alta_geo = gpd.GeoDataFrame(geometry=[buffer_inundacion_alta_m], crs=CRS_METRICO).to_crs(CRS_MAPA)
buffer_mod_geo = gpd.GeoDataFrame(geometry=[buffer_inundacion_mod_m], crs=CRS_METRICO).to_crs(CRS_MAPA)

# Intersección Vial Estricta
vial_expuesto_m = vial_filtrado_m.loc[
    vial_filtrado_m.intersects(buffer_geom_m) & vial_filtrado_m.intersects(geom_remocion_m)
]
km_vial_expuesto = vial_expuesto_m.geometry.length.sum() / 1000
km_vial_total = vial_filtrado_m.geometry.length.sum() / 1000

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("GeoVisualizador de Vulnerabilidad — Cuenca de Corral")
st.caption("Haz clic sobre cualquier punto del mapa para consultar la elevación y trazar perfiles.")

# ----------------------------------------------------------------------------
# CONSTRUCCIÓN DEL MAPA (Ancho Completo)
# ----------------------------------------------------------------------------
TILES = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Positron": "CartoDB positron",
    "CartoDB Dark": "CartoDB dark_matter",
}

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

if mostrar_cuenca:
    folium.GeoJson(
        capas_geo["cuenca"],
        name="Límite de cuenca",
        style_function=lambda x: {"fillOpacity": 0, "color": "#2c3e50", "weight": 3, "dashArray": "6,4"},
    ).add_to(m)

if mostrar_dem:
    img_dem, b_dem = array_a_png_overlay(dem, plt.cm.terrain, np.nanmin(dem), np.nanmax(dem), dem_bounds)
    folium.raster_layers.ImageOverlay(image=img_a_data_url(img_dem), bounds=b_dem, opacity=0.6, name="DEM").add_to(m)

if mostrar_pendiente:
    # Nuevo colormap contrastante: Verde, Amarillo, Naranja oscuro, Rojo Oscuro
    cmap_pend = ListedColormap(["#00FF00", "#FFFF00", "#FF8C00", "#8B0000"])
    img, b = array_a_png_overlay(clases_pendiente, cmap_pend, 0, 3, dem_bounds)
    folium.raster_layers.ImageOverlay(
        image=img_a_data_url(img), bounds=b, opacity=0.65, name="Susceptibilidad a remoción en masa"
    ).add_to(m)

if mostrar_buffer:
    folium.GeoJson(
        buffer_mod_geo, name=f"Inundación Riesgo Moderado",
        style_function=lambda x: {"fillColor": "#3498db", "color": "#2980b9", "weight": 1, "fillOpacity": 0.25},
    ).add_to(m)
    folium.GeoJson(
        buffer_alta_geo, name=f"Inundación Riesgo Alto (<5° pend.)",
        style_function=lambda x: {"fillColor": "#1abc9c", "color": "#16a085", "weight": 1, "fillOpacity": 0.6},
    ).add_to(m)

if mostrar_rios and not rios_filtrado_geo.empty:
    folium.GeoJson(
        rios_filtrado_geo, name="Red hidrográfica",
        style_function=lambda x: {"color": "#2980b9", "weight": 1 + (x["properties"].get("strahler_n") or 1), "opacity": 0.9}
    ).add_to(m)

if mostrar_vial and not vial_filtrado_geo.empty:
    folium.GeoJson(
        vial_filtrado_geo, name="Red vial",
        style_function=lambda x: {"color": "#34495e", "weight": 2.5, "opacity": 0.8}
    ).add_to(m)

if mostrar_poblacion:
    poblacion_geo = capas_geo["poblacion"]
    if modo_poblacion == "Categoría (Aldea/Pueblo)":
        def estilo_pob(f): return {"fillColor": "#e67e22" if f["properties"].get("CATEGORIA") == "Pueblo" else "#27ae60", "color": "#2c3e50", "weight": 1.5, "fillOpacity": 0.5}
    else:
        cmap_vuln = plt.cm.YlOrRd
        norm_vuln = Normalize(0, 100)
        def estilo_pob(f):
            r, g, b_, _ = cmap_vuln(norm_vuln(f["properties"].get("indice_vulnerabilidad") or 0))
            return {"fillColor": f"#{int(r*255):02x}{int(g*255):02x}{int(b_*255):02x}", "color": "#2c3e50", "weight": 1.5, "fillOpacity": 0.75}

    folium.GeoJson(
        poblacion_geo, name="Localidades",
        style_function=estilo_pob,
        tooltip=folium.GeoJsonTooltip(["NOMBRE_LOC", "indice_vulnerabilidad"])
    ).add_to(m)

    # Nombres siempre visibles encima
    for _, row in poblacion_geo.iterrows():
        if pd.notnull(row["NOMBRE_LOC"]):
            lbl = f'<div style="font-size: 11pt; color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold; white-space: nowrap;">{row["NOMBRE_LOC"]}</div>'
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                icon=folium.DivIcon(html=lbl, icon_anchor=(0, 0))
            ).add_to(m)

MiniMap(toggle_display=True).add_to(m)
Fullscreen().add_to(m)
MeasureControl(primary_length_unit="meters").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

map_data = st_folium(m, use_container_width=True, height=550, returned_objects=["last_clicked"])

# ----------------------------------------------------------------------------
# PANEL INFERIOR Y GRÁFICOS
# ----------------------------------------------------------------------------
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    nuevo_punto = (lat, lon)
    
    with rasterio.open(RUTAS["dem"]) as src:
        val_gen = src.sample([(lon, lat)])
        elev_val = list(val_gen)[0][0]
        elev_txt = f"{round(float(elev_val), 1)} m" if elev_val != src.nodata else "Sin datos"
            
    st.info(f"**Elevación en punto clicado:** {elev_txt} s.n.m. (Lat: {lat:.4f}, Lon: {lon:.4f})")
    
    if not st.session_state.puntos_perfil or st.session_state.puntos_perfil[-1] != nuevo_punto:
        st.session_state.puntos_perfil.append(nuevo_punto)
        if len(st.session_state.puntos_perfil) > 2:
            st.session_state.puntos_perfil.pop(0)

col_izq, col_der = st.columns(2)

with col_izq:
    st.markdown("### Índice de Vulnerabilidad por Localidad")
    fig, ax = plt.subplots(figsize=(6, 3))
    datos_barra = capas_m["poblacion"].sort_values("indice_vulnerabilidad")
    colores = plt.cm.YlOrRd(Normalize(0, 100)(datos_barra["indice_vulnerabilidad"]))
    ax.barh(datos_barra["NOMBRE_LOC"], datos_barra["indice_vulnerabilidad"], color=colores)
    ax.set_xlabel("Vulnerabilidad Espacial Estructurada (0-100)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(f"**Vialidad en riesgo crítico:** {km_vial_expuesto:.1f} km expuestos simultáneamente a inundación y remoción.")

with col_der:
    st.markdown("###  Perfil de elevación")
    st.caption("El perfil se generará automáticamente con tus dos últimos clics en el mapa.")
    
    if len(st.session_state.puntos_perfil) == 2:
        lat_a, lon_a = st.session_state.puntos_perfil[0]
        lat_b, lon_b = st.session_state.puntos_perfil[1]
        
        n_muestras = 150
        lats = np.linspace(lat_a, lat_b, n_muestras)
        lons = np.linspace(lon_a, lon_b, n_muestras)

        with rasterio.open(RUTAS["dem"]) as src:
            valores = list(src.sample(zip(lons, lats)))
            elevaciones = [v[0] if v[0] != src.nodata else np.nan for v in valores]

        punto_a = gpd.GeoSeries([Point(lon_a, lat_a)], crs=CRS_MAPA).to_crs(CRS_METRICO)
        punto_b = gpd.GeoSeries([Point(lon_b, lat_b)], crs=CRS_MAPA).to_crs(CRS_METRICO)
        distancia = punto_a.distance(punto_b).iloc[0] / 1000
        eje_x = np.linspace(0, distancia, n_muestras)

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.fill_between(eje_x, elevaciones, color="#7f8c8d", alpha=0.4)
        ax2.plot(eje_x, elevaciones, color="#2c3e50", linewidth=1.5)
        ax2.set_xlabel("Distancia (km)")
        ax2.set_ylabel("Elevación (m)")
        ax2.set_title("Corte Transversal Topográfico")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
        
        if st.button("Limpiar perfil"):
            st.session_state.puntos_perfil = []
            st.rerun()
    else:
        st.info("Haz clic en dos puntos diferentes del mapa para generar el perfil de elevación.")