import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Cargar datos (asegúrate de que la ruta sea correcta para tu archivo)
df = pd.read_csv("C:\\TFG\\margen_tfg.csv")

# Limpieza y preparación de datos
df['precio_total'] = df['precio_total'].astype(str).str.replace(',', '.').astype(float)
df['margen'] = df['margen'].astype(str).str.replace(',', '.').astype(float)
df['total_ventas'] = pd.to_numeric(df['total_ventas'])

# Calcular beneficios
df['beneficio_unidad'] = df['precio_total'] * df['margen']
df['beneficio_total'] = df['beneficio_unidad'] * df['total_ventas']

# Calcular Media y Desviación Estándar de los beneficios totales
mu = df['beneficio_total'].mean()
sigma = df['beneficio_total'].std()

# Umbral ±10% de la media
umbral = mu * 0.1

# Filtrar productos candidatos dentro del rango del ±10% de la media
df_candidatos = df[(df['beneficio_total'] >= (mu - umbral)) & (df['beneficio_total'] <= (mu + umbral))]

# Seleccionar el producto con el mayor beneficio total dentro de ese rango
if not df_candidatos.empty:
    producto_seleccionado = df_candidatos.loc[df_candidatos['beneficio_total'].idxmax()]
    print("Producto seleccionado:")
    print(producto_seleccionado[['codigo_modelo_codificado', 'precio_total', 'margen', 'total_ventas', 'beneficio_total']])
else:
    # Fallback si no hay candidatos. Esto asegura que 'producto_seleccionado' siempre exista.
    producto_seleccionado = {'beneficio_codificado': 'N/A', 'precio_total': 0, 'margen': 0, 'total_ventas': 0, 'beneficio_total': mu}
    print("No se encontraron productos dentro del umbral ±10% de la media. Usando la media como referencia para el producto seleccionado en el gráfico.")


# --- Configuración del Gráfico ---
plt.figure(figsize=(14, 8)) # Tamaño del gráfico para buena visibilidad
sns.set_style('whitegrid') # Estilo de fondo

# --- Definir el rango del eje X ---
x_min_display = 0
x_max_fixed = 10000000 # <-- Modificado a 10,000,000

# El x_min final para el linspace y el plot será el máximo entre 0 y mu - 4*sigma
# Esto asegura que si la cola izquierda de la normal es negativa, empezamos en 0.
x_final_min = max(x_min_display, mu - 4 * sigma)
# El x_max final será el valor fijo de 10,000,000
x_final_max = x_max_fixed


# Crear puntos para la curva de la Distribución Normal en el rango ajustado
x = np.linspace(x_final_min, x_final_max, 500) # 500 puntos para una curva suave
pdf = norm.pdf(x, mu, sigma) # Calcular la Densidad de Probabilidad para cada punto

# --- Trazado del Área Bajo la Curva y la Curva Normal ---
plt.fill_between(x, 0, pdf, color='skyblue', alpha=0.2, label='Área de Densidad') # Área azul claro con más transparencia
plt.plot(x, pdf, color='blue', linestyle='-', linewidth=2, label='Distribución Normal Ajustada', zorder=2) # Curva azul, encima del área

# --- Trazado de las Líneas Verticales Clave ---
# Ajustar linewidth y zorder para que se vean mejor si se superponen
plt.axvline(mu, color='darkblue', linestyle='--', linewidth=1.75, label=f'Media ({mu:,.0f} €)', zorder=3)
plt.axvline(mu - umbral, color='red', linestyle=':', linewidth=1.5, label=f'-10% de la Media ({mu - umbral:,.0f} €)', zorder=4)
plt.axvline(mu + umbral, color='red', linestyle=':', linewidth=1.5, label=f'+10% de la Media ({mu + umbral:,.0f} €)', zorder=4)

# Solo dibujar la línea del producto seleccionado si su beneficio es relevante y está dentro del rango visible.
# Priorizar la línea del producto seleccionado con un zorder más alto
if x_final_min <= producto_seleccionado['beneficio_total'] <= x_final_max:
    plt.axvline(producto_seleccionado['beneficio_total'], color='green', linestyle='-', linewidth=1.75,
                label=f'Producto Seleccionado ({producto_seleccionado["beneficio_total"]:,.0f} €)', zorder=5)
else:
    print(f"Nota: El producto seleccionado ({producto_seleccionado['beneficio_total']:,.0f} €) está fuera del rango visible del gráfico ({x_final_min:,.0f} € a {x_final_max:,.0f} €).")


# --- Configuración del Gráfico (Estética) ---
plt.title("Distribución de Beneficio Total", fontsize=16)
plt.xlabel("Beneficio Total (€)", fontsize=13)
plt.ylabel("Densidad de Probabilidad", fontsize=13)
plt.xlim(x_final_min, x_final_max) # Aplicar el rango
plt.ticklabel_format(style='plain', axis='x') # Evitar notación científica en el eje X
plt.ylim(bottom=0) # Asegurar que el eje Y empiece en 0

# Leyenda DENTRO del gráfico
plt.legend(loc='upper right') # Ubicación común y que suele funcionar bien

plt.tight_layout() # Eliminar rect para que la leyenda pueda autoajustarse dentro
plt.show()