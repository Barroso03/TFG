import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# --- Cargar datos desde CSV ---
df = pd.read_csv(r'C:\TFG\SERIES\todas_las_marcas.csv')  # Cambia esta ruta si hace falta
df['Week'] = pd.to_datetime(df['Week'])
df.set_index('Week', inplace=True)

# --- Asegurar valores numéricos ---
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# --- Aplicar medias móviles de 4 semanas ---
df_ma = df.rolling(window=4, center=True).mean()
df_ma.dropna(inplace=True)

# --- Normalizar las series suavizadas ---
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_ma), columns=df_ma.columns, index=df_ma.index)

# --- Graficar las series normalizadas y suavizadas ---
plt.figure(figsize=(14, 7))
for col in df_norm.columns:
    plt.plot(df_norm.index, df_norm[col], label=col)
plt.title('Series de Google Trends suavizadas (media móvil 4 semanas) y normalizadas')
plt.xlabel('Semana')
plt.ylabel('Valor normalizado')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Calcular matriz de correlación ---
corr_matrix = df_norm.corr()

# --- Mostrar matriz en consola ---
print("Matriz de correlación (medias móviles de 4 semanas, normalizadas):\n")
print(corr_matrix)

# --- Visualizar la matriz como heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlación entre marcas (media móvil de 4 semanas, normalizadas)")
plt.tight_layout()
plt.show()
