import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

# --- Configuración general ---
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

print("--- INICIO DEL PROCESO ---")

# --- Paso 1: Cargar y preparar datos ---
print("\n[Paso 1/4] Cargando y preprocesando datos (agrupación semanal)...")
df_trends = pd.read_csv(r'C:\TFG\SERIES\google_trends.csv')
df_trafico = pd.read_csv(r'C:\TFG\SERIES\trafico.csv', sep=';', parse_dates=['fecha_oferta_dia'])

df_trends['Week'] = pd.to_datetime(df_trends['Week'])
df_trafico['fecha_oferta_dia'] = pd.to_datetime(df_trafico['fecha_oferta_dia'])

df_trafico['Week'] = df_trafico['fecha_oferta_dia'] - pd.to_timedelta((df_trafico['fecha_oferta_dia'].dt.dayofweek + 1) % 7, unit='d')
df_trafico_semanal = df_trafico.groupby('Week', as_index=False)['trafico'].sum()

df = pd.merge(df_trends, df_trafico_semanal, on='Week', how='inner')
df.set_index('Week', inplace=True)

df['Mimarca'] = pd.to_numeric(df['Mimarca'], errors='coerce')
df['trafico'] = pd.to_numeric(df['trafico'], errors='coerce')
df.dropna(inplace=True)

print(f"   Datos preprocesados. Rango: {df.index.min()} a {df.index.max()} — Total semanas: {len(df)}")

# --- Visualización Inicial: Series normalizadas sin transformación ---
print("\n[Visualización] Series originales normalizadas sin procesamiento adicional...")

scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[['Mimarca_norm', 'trafico_norm']] = scaler.fit_transform(df_norm[['Mimarca', 'trafico']])

plt.figure(figsize=(14, 6))
plt.plot(df_norm.index, df_norm['Mimarca_norm'], label='Tráfico concesionario (Normalizado)', color='blue')
plt.plot(df_norm.index, df_norm['trafico_norm'], label='Google trends (Normalizado)', color='red', linestyle='--')
plt.title('Series Originales Normalizadas: Google Trends vs Tráfico Web')
plt.xlabel('Semana')
plt.ylabel('Valores Normalizados (0–1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Paso 2: Calcular correlaciones ---
print("\n[Paso 2/4] Calculando correlaciones para ventanas de medias móviles y desplazamientos...")

resultados = []

for ventana in range(1, 7):  # Medias móviles de 1 a 6 semanas
    df_tmp = df.copy()
    df_tmp['Mimarca_ma'] = df_tmp['Mimarca'].rolling(window=ventana, center=True).mean()
    df_tmp['trafico_ma'] = df_tmp['trafico'].rolling(window=ventana, center=True).mean()
    df_tmp = df_tmp.dropna(subset=['Mimarca_ma', 'trafico_ma']).copy()

    scaler = MinMaxScaler()
    df_tmp['Mimarca_scaled'] = scaler.fit_transform(df_tmp[['Mimarca_ma']])
    df_tmp['trafico_scaled'] = scaler.fit_transform(df_tmp[['trafico_ma']])

    for lag in range(1, 31):  # Desplazamientos de 1 a 30 semanas
        df_tmp['trafico_shifted'] = df_tmp['trafico_scaled'].shift(lag)
        df_corr = df_tmp.dropna(subset=['Mimarca_scaled', 'trafico_shifted'])

        if len(df_corr) > 10:
            corr = df_corr['Mimarca_scaled'].corr(df_corr['trafico_shifted'])
            resultados.append({
                'ventana': ventana,
                'lag': lag,
                'correlacion': corr
            })

# --- Paso 3: Mostrar top 3 correlaciones ---
print("\n[Paso 3/4] Top 3 combinaciones con mayor correlación:\n")
df_resultados = pd.DataFrame(resultados).sort_values(by='correlacion', ascending=False)
top_3 = df_resultados.head(3)
print(top_3.to_string(index=False))

# --- Paso 4: Visualizar mejor correlación ---
print("\n[Paso 4/4] Pintando la mejor correlación encontrada...")

mejor = top_3.iloc[0]
ventana = int(mejor['ventana'])
lag = int(mejor['lag'])

df_tmp = df.copy()
df_tmp['Mimarca_ma'] = df_tmp['Mimarca'].rolling(window=ventana, center=True).mean()
df_tmp['trafico_ma'] = df_tmp['trafico'].rolling(window=ventana, center=True).mean()
df_tmp = df_tmp.dropna(subset=['Mimarca_ma', 'trafico_ma']).copy()

scaler = MinMaxScaler()
df_tmp['Mimarca_scaled'] = scaler.fit_transform(df_tmp[['Mimarca_ma']])
df_tmp['trafico_scaled'] = scaler.fit_transform(df_tmp[['trafico_ma']])
df_tmp['trafico_shifted'] = df_tmp['trafico_scaled'].shift(lag)

df_plot = df_tmp.dropna(subset=['Mimarca_scaled', 'trafico_shifted'])

plt.figure(figsize=(14, 6))
plt.plot(df_plot.index, df_plot['Mimarca_scaled'], label='Google Trends (Escalado)', color='blue')
plt.plot(df_plot.index, df_plot['trafico_shifted'], label=f'Tráfico concesionario (Escalado y desplazado {lag} semanas)', color='red', linestyle='--')
plt.title(f'Mejor Correlación — Ventana: {ventana} | Lag: {lag} | Corr: {mejor["correlacion"]:.4f}')
plt.xlabel('Semana')
plt.ylabel('Series Escaladas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- FIN DEL PROCESO ---")
