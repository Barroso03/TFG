import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Matriz de confusión (valores absolutos)
conf_matrix = np.array([
    [1125, 208],   # No tiene tráfico
    [142, 265]    # Tiene tráfico
])

# Etiquetas de filas y columnas
labels = ['No ha comprado', 'Ha comprado']
columns = ['Pred No ha comprado', 'Pred ha comprado']

# Calcular porcentajes por fila
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

# Crear anotaciones con el símbolo %
annot = np.array([[f"{val:.1f}%" for val in row] for row in conf_matrix_percent])

# Dibujar heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_percent, annot=annot, fmt='', cmap='Blues',
            xticklabels=columns, yticklabels=labels, cbar_kws={'label': 'Porcentaje (%)'})

plt.title('Matriz de confusión (porcentajes por clase real)')
plt.ylabel('Clase real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()

