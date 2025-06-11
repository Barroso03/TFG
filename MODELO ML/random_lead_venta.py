# ========================
# IMPORTACIÓN DE LIBRERÍAS
# ========================
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)

# ========================
# CONEXIÓN A SQL SERVER
# ========================
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=localhost\\SQLEXPRESS;'
    'DATABASE=coches;'
    'Trusted_Connection=yes;'
    'Encrypt=no;'
)

# ========================
# CONSULTA SQL Y CARGA
# ========================
sql_query = """
    SELECT	
        wp.showroom, wp.cc,
        wl.organic_search, wl.paid_search, wl.channel_max, wl.dif_days, wl.campaign_cod,
        cc.config,
        l2.codigo_concesionario_origen, 
        m.Max_Mosaic_G,
        t.codigo_tienda, t.codigo_exposicion, t.codigo_producto, t.cod_oferta,
        t.identificador_oferta, t.origen_contacto, t.origen_negociacion, 
        t.vendedor_codificado_final, t.PRODUCT_CODE_TRAFICO, t.total_trafico, t.ha_comprado
    FROM  stg_l l2
    LEFT JOIN  stg_t t  on l2.id_trafico = t.id_trafico
    LEFT JOIN web_lead_mod wl ON l2.id_web = wl.id_web
    LEFT JOIN web_path wp ON wl.id_web_path = wp.id_web_path 
    LEFT JOIN mosaic m ON l2.id_mosaic = m.id_mosaic
    LEFT JOIN CC_mod cc ON cc.id_CC_mod = wl.id_CC_mod
    WHERE Max_Mosaic_G IS NOT NULL AND tiene_trafico = 1
"""


df = pd.read_sql(sql_query, conn)
conn.close()

# ========================
# LIMPIEZA Y PREPROCESAMIENTO
# ========================
# Rellenar valores nulos
# Rellenar valores nulos solo en columnas que realmente existen tras la consulta SQL
columnas_a_rellenar = [
    "organic_search", "paid_search", "config"
]
df[columnas_a_rellenar] = df[columnas_a_rellenar].fillna(0)





# ========================
# AGRUPACIONES PERSONALIZADAS
# ========================
df['dif_days'] = pd.to_numeric(df['dif_days'], errors='coerce')
df['dif_days_group'] = df['dif_days'].apply(lambda d: 0 if d <= 0 else 1 if d <= 7 else 2 if d <= 30 else 3 if d <= 90 else 4 if d <= 120 else 5)
df.drop(columns=['dif_days'], inplace=True)


df['paid_search_group'] = df['paid_search'].apply(lambda p: 0 if p <= 0 else 1 if p <= 5 else 2 if p <= 10 else 3)
df.drop(columns=['paid_search'], inplace=True)

# ========================
# MODELADO CON RANDOM FOREST
# ========================
X = df.drop(['ha_comprado'], axis=1, errors='ignore')
y = df['ha_comprado']

X.fillna(0, inplace=True)

# Codificación
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:  # Solo columnas de tipo 'object'
    # Asegurarse de que la columna tenga tipo 'str' (o 'int' si es necesario)
    X[col] = X[col].astype(str)  # o X[col] = X[col].astype(int)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [300, 400, 500],
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt'],
}

results = []
for params in product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    model = RandomForestClassifier(**param_dict, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None
    cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    train_score = model.score(X_train, y_train)

    overfit = "SEVERE OVERFITTING" if train_score > 1.5 * cv_score else "DANGER" if train_score > 1.2 * cv_score else "OK"

    results.append({**param_dict, 'accuracy': accuracy, 'f1_score': f1, 'recall': recall,
                    'precision': precision, 'roc_auc': roc_auc, 'cv_score': cv_score,
                    'train_score': train_score, 'overfitting': overfit})

results_df = pd.DataFrame(results)
results_df.to_csv(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\hiperparametros_modelo_lead_venta.csv', index=False)

# ========================
# MEJOR MODELO Y EXPORTACIÓN PARA POWER BI
# ========================

# ========================
# MEJOR MODELO RANDOM FOREST (sin overfitting)
# ========================
mejor = results_df[results_df['overfitting'] == 'OK'].sort_values(by='f1_score', ascending=False).iloc[0]

print("Mejores hiperparámetros del modelo seleccionado:")
for k in param_grid:
    print(f"{k}: {mejor[k]}")

# Entrenar el modelo con los mejores hiperparámetros
mejor_modelo = RandomForestClassifier(**{k: mejor[k] for k in param_grid}, random_state=42)
mejor_modelo.fit(X_train, y_train)

# ========================
# PREDICCIONES Y MÉTRICAS
# ========================
y_pred = mejor_modelo.predict(X_test)
y_pred_proba = mejor_modelo.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nMétricas del modelo entrenado:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# ========================
# CURVA ROC
# ========================
y_pred_proba = mejor_modelo.predict_proba(X_test)[:, 1]
print(y_pred_proba)
print(y_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Guardar el archivo CSV con ; como delimitador y comas como separador decimal
roc_data = pd.DataFrame({
    'y_test': y_test,
    'y_pred_proba': y_pred_proba
})

# Guardar el CSV en la ubicación deseada
roc_data.to_csv(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\predicciones_venta.csv', 
                sep=';',   # Separador de columnas por punto y coma
                index=False, 
                decimal=',')  # Usar coma como separador decimal





# También guardar imagen opcionalmente
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_pred_proba):.2f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.savefig(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\roc_curve_lead_venta.png')
plt.show()

# ========================
# MATRIZ DE CONFUSIÓN
# ========================
y_pred = mejor_modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Guardar matriz para Power BI
cm_df = pd.DataFrame(cm, columns=['Pred No ha comprado', 'Pred ha comprado'], index=['No ha comprado', 'Ha comprado'])
cm_df.reset_index(inplace=True)
cm_df.rename(columns={'index': 'Actual'}, inplace=True)
cm_df.to_csv(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\confusion_matrix_lead_venta.csv', index=False)

# También guardar heatmap opcional
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.savefig(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\matriz_confusion_lead_venta.png')
plt.show()

# ========================
# IMPORTANCIA DE VARIABLES
# ========================
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mejor_modelo.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Guardar para Power BI
importances.to_csv(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\importancia_variables_lead_venta.csv', index=False)

# También guardar gráfico opcional
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importances.head(10))
plt.title("Top 10 Variables Más Importantes")
plt.savefig(r'C:\\TFG\\MODELO ML\\RESULTADOS\\LEAD VENTA\\importancia_variables_lead_venta.png')
plt.show()

