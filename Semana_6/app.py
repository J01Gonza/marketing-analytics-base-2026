# App de segmentación de clientes: visor de grupos + comparador de técnicas
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

NUM = ['ingreso_anual', 'gasto_total', 'compras_tienda', 'compras_web', 'compras_catalogo',
       'visitas_web_mes', 'recencia_dias', 'hijos_pequenos', 'edad']
CAT = ['educacion', 'estado_civil']

st.set_page_config(page_title='Segmentación de clientes', layout='wide')
st.title('Segmentación de clientes')


def limpiar(df):
    # Orange antepone el tipo a la columna (m#id_cliente, D#educacion); se quita el prefijo
    df = df.rename(columns=lambda col: col.split('#')[-1])
    df = df.rename(columns={'respondio_ultima_campana': 'respondio'})
    # Si el CSV trae filas extra de tipos (formato de Orange), se descartan
    if 'gasto_total' in df:
        df = df[pd.to_numeric(df['gasto_total'], errors='coerce').notna()].copy()
        for col in NUM + ['respondio']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def perfil(df, col):
    g = df.groupby(col)
    t = pd.DataFrame({
        'clientes': g.size(),
        '% clientes': (g.size() / len(df) * 100).round(1),
        'gasto promedio': g['gasto_total'].mean().round(0),
        '% del gasto': (g['gasto_total'].sum() / df['gasto_total'].sum() * 100).round(1),
    })
    if 'ingreso_anual' in df:
        t['ingreso promedio'] = g['ingreso_anual'].mean().round(0)
    if 'respondio' in df:
        t['% respondió campaña'] = (g['respondio'].mean() * 100).round(1)
    return t.sort_values('gasto promedio', ascending=False)


tab_visor, tab_comp = st.tabs(['Visor de grupos', 'Comparador de técnicas'])

# ---------------- Visor de grupos ----------------
with tab_visor:
    st.write('Suba el CSV guardado desde Orange (`clientes_con_cluster.csv`) o el del notebook (`segmentos_notebook.csv`).')
    archivo = st.file_uploader('CSV con una columna de grupos', type='csv', key='visor')
    if archivo is not None:
        df = limpiar(pd.read_csv(archivo))
        candidatas = [c for c in df.columns if c == 'Cluster' or c.startswith('seg_')]
        if not candidatas:
            candidatas = list(df.columns)
        col_grupo = st.selectbox('Columna de grupos', candidatas)
        df[col_grupo] = df[col_grupo].astype(str)

        numericas = [c for c in NUM if c in df]
        c1, c2 = st.columns(2)
        x = c1.selectbox('Eje x', numericas, index=numericas.index('ingreso_anual') if 'ingreso_anual' in numericas else 0)
        y = c2.selectbox('Eje y', numericas, index=numericas.index('gasto_total') if 'gasto_total' in numericas else 0)
        fig = px.scatter(df, x=x, y=y, color=col_grupo, opacity=0.6,
                         category_orders={col_grupo: sorted(df[col_grupo].unique())})
        st.plotly_chart(fig, width='stretch')

        st.subheader('Perfil por grupo')
        st.dataframe(perfil(df, col_grupo), width='stretch')

# ---------------- Comparador de técnicas ----------------
with tab_comp:
    st.write('Ajusta las cinco técnicas con las mismas variables estandarizadas '
             '(k-prototypes agrega educación y estado civil).')
    archivo2 = st.file_uploader('CSV de clientes (por defecto `clientes_retail.csv`)', type='csv', key='comp')
    try:
        base = limpiar(pd.read_csv(archivo2 if archivo2 is not None else 'clientes_retail.csv'))
    except FileNotFoundError:
        st.warning('Suba clientes_retail.csv para continuar.')
        st.stop()

    c1, c2, c3 = st.columns(3)
    k = c1.slider('Número de grupos (k)', 2, 8, 3)
    eps = c2.slider('eps de DBSCAN', 0.5, 3.0, 1.5, 0.1)
    min_vec = c3.slider('Mínimo de vecinos (DBSCAN)', 3, 20, 5)

    @st.cache_data
    def ajustar(df, k, eps, min_vec):
        Z = StandardScaler().fit_transform(df[NUM])
        res = {
            'k-means': KMeans(k, n_init=10, random_state=0).fit_predict(Z),
            'Jerárquico (Ward)': AgglomerativeClustering(k, linkage='ward').fit_predict(Z),
            'DBSCAN': DBSCAN(eps=eps, min_samples=min_vec).fit_predict(Z),
            'Mezcla gaussiana': GaussianMixture(k, random_state=0, n_init=5).fit_predict(Z),
        }
        if all(c in df for c in CAT):
            X = np.column_stack([Z, df[CAT].astype(str).values]).astype(object)
            kp = KPrototypes(n_clusters=k, init='Cao', random_state=0, n_init=3)
            res['k-prototypes'] = kp.fit_predict(X, categorical=[len(NUM), len(NUM) + 1])
        return Z, res

    with st.spinner('Ajustando modelos...'):
        Z, res = ajustar(base, k, eps, min_vec)

    filas = []
    for nombre, etiquetas in res.items():
        grupos = sorted(set(etiquetas) - {-1})
        ruido = int((etiquetas == -1).sum())
        mask = etiquetas != -1
        sil = silhouette_score(Z[mask], etiquetas[mask]) if len(grupos) > 1 else np.nan
        tamanos = ', '.join(str(int((etiquetas == g).sum())) for g in grupos[:10])
        filas.append({'técnica': nombre, 'grupos': len(grupos), 'ruido': ruido,
                      'tamaños': tamanos + (' ...' if len(grupos) > 10 else ''),
                      'silueta': round(sil, 3)})
    st.subheader('Resumen por técnica')
    st.dataframe(pd.DataFrame(filas).set_index('técnica'), width='stretch')

    st.subheader('Coincidencia entre técnicas (Adjusted Rand Index, 1 = idénticas)')
    nombres = list(res)
    ari = pd.DataFrame(np.eye(len(nombres)), index=nombres, columns=nombres)
    for a, b in itertools.combinations(nombres, 2):
        ari.loc[a, b] = ari.loc[b, a] = adjusted_rand_score(res[a], res[b])
    st.plotly_chart(px.imshow(ari.round(2), text_auto=True, zmin=0, zmax=1,
                              color_continuous_scale='Blues'), width='stretch')

    st.subheader('Ver una técnica')
    elegida = st.selectbox('Técnica', nombres)
    vista = base.copy()
    vista['grupo'] = pd.Series(res[elegida], index=vista.index).astype(str).replace('-1', 'ruido')
    st.plotly_chart(px.scatter(vista, x='ingreso_anual', y='gasto_total', color='grupo', opacity=0.6),
                    width='stretch')
    st.dataframe(perfil(vista, 'grupo'), width='stretch')
