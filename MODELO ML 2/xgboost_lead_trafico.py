# ========================
# IMPORTACIÓN DE LIBRERÍAS
# ========================
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
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
        wp.offers, wp.Home, wp.showroom, wp.promotions, wp.others, wp.cc, wp.DLR, wp.FLEET, wp.PV,
        wl.organic_search, 
        wl.paid_search, wl.channel_max, wl.dif_days,wl.campaign_cod,
        cc.config,
        l2.codigo_concesionario_origen, 
        l2.tiene_trafico,
        m.Max_Mosaic_G
    FROM stg_l l2
    LEFT JOIN web_lead_mod wl ON l2.id_web = wl.id_web
    LEFT JOIN Coste_cd_mod ccm ON wl.id_Coste_cd_mod = ccm.id_Coste_cd_mod
    LEFT JOIN web_path wp ON wl.id_web_path = wp.id_web_path 
    LEFT JOIN mosaic m ON l2.id_mosaic = m.id_mosaic
    LEFT JOIN CC_mod cc ON cc.id_CC_mod = wl.id_CC_mod
    WHERE Max_Mosaic_G IS NOT NULL
"""

df = pd.read_sql(sql_query, conn)
conn.close()

# ========================
# LIMPIEZA Y PREPROCESAMIENTO
# ========================
columnas_a_rellenar = [
    "organic_search", "paid_search","config"
]


df[columnas_a_rellenar] = df[columnas_a_rellenar].fillna(0)



# Crear columna 'total' para filtrar
cols_interaccion = ['offers', 'Home', 'showroom', 'promotions', 'others', 'cc', 'DLR', 'FLEET', 'PV']
df['total'] = df[cols_interaccion].sum(axis=1)

# Filtrar comportamiento sospechoso
condicion_filtrado = (df['total'] > 5) & (df['tiene_trafico'] == 0) & (df['paid_search'] == 0)
df = df[~condicion_filtrado]
df.drop(columns=['total'], inplace=True)


# Eliminar columnas irrelevantes
columnas_a_eliminar = [
    'offers', 'DLR', 'FLEET', 'PV', 'promotions', 'others',"Home"
]

df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

# ========================
# AGRUPACIONES PERSONALIZADAS
# ========================
df['dif_days'] = pd.to_numeric(df['dif_days'], errors='coerce')
df['dif_days_group'] = df['dif_days'].apply(lambda d: 0 if d <= 0 else 1 if d <= 7 else 2 if d <= 30 else 3 if d <= 90 else 4 if d <= 120 else 5)
df.drop(columns=['dif_days'], inplace=True)


df['paid_search_group'] = df['paid_search'].apply(lambda p: 0 if p <= 0 else 1 if p <= 5 else 2 if p <= 10 else 3)
df.drop(columns=['paid_search'], inplace=True)


# ========================
# MODELADO CON XGBOOST
# ========================
X = df.drop(['tiene_trafico'], axis=1, errors='ignore')
y = df['tiene_trafico'].astype(int)  # XGBoost requiere int

X.fillna(0, inplace=True)

# Codificación
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.001],
    'subsample': [0.6,0.8, 1.0],
    'colsample_bytree': [0.6,0.8, 1.0],
    'scale_pos_weight': [1, 2],  # útil si hay desbalance
    'gamma': [0, 1]
}

results = []
for params in product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    model = XGBClassifier(**param_dict, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    train_score = model.score(X_train, y_train)

    overfit = "SEVERE OVERFITTING" if train_score > 1.5 * cv_score else "DANGER" if train_score > 1.2 * cv_score else "OK"

    results.append({**param_dict, 'accuracy': accuracy, 'f1_score': f1, 'recall': recall,
                    'precision': precision, 'roc_auc': roc_auc, 'cv_score': cv_score,
                    'train_score': train_score, 'overfitting': overfit})

results_df = pd.DataFrame(results)
results_df.to_csv(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\hiperparametros_modelo_lead_trafico.csv', index=False)

# ========================
# MEJOR MODELO Y EXPORTACIÓN PARA POWER BI
# ========================
# ========================
# MEJOR MODELO POR F1_SCORE
# ========================
mejor = results_df.sort_values(by='f1_score', ascending=False).iloc[0]

print("Mejores hiperparámetros del mejor modelo por F1 Score:")
for k in param_grid:
    print(f"{k}: {mejor[k]}")

# Entrenamiento del modelo
mejor_modelo = XGBClassifier(**{k: mejor[k] for k in param_grid}, random_state=42, use_label_encoder=False, eval_metric='logloss')
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

print("\nMétricas del mejor modelo (evaluado sobre test):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# ========================
# CURVA ROC
# ========================
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

roc_data = pd.DataFrame({'y_test': y_test, 'y_pred_proba': y_pred_proba})
roc_data.to_csv(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\predicciones_trafico.csv',
                sep=';', index=False, decimal=',')

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.savefig(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\roc_curve_lead_trafico.png')
plt.show()

# ========================
# MATRIZ DE CONFUSIÓN
# ========================
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, columns=['Pred No tiene trafico', 'Pred Tiene trafico'], index=['No tiene trafico', 'Tiene trafico'])
cm_df.reset_index(inplace=True)
cm_df.rename(columns={'index': 'Actual'}, inplace=True)
cm_df.to_csv(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\confusion_matrix_lead_trafico.csv', index=False)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.savefig(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\matriz_confusion_lead_trafico.png')
plt.show()

# ========================
# IMPORTANCIA DE VARIABLES
# ========================
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mejor_modelo.feature_importances_
}).sort_values(by='Importance', ascending=False)

importances.to_csv(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\importancia_variables_lead_trafico.csv', index=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importances.head(10))
plt.title("Top 10 Variables Más Importantes")
plt.savefig(r'C:\\TFG\\MODELO ML 2\\RESULTADOS\\LEAD TRAFICO\\importancia_variables_lead_trafico.png')
plt.show()
