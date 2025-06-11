import pyodbc
import pandas as pd
import numpy as np

# 📌 🔹 Conexión a **Azure SQL** (Autenticación AAD Interactiva)
AZURE_SERVER = 'uaxmathfis.database.windows.net'
AZURE_DATABASE = 'rd'
AZURE_DRIVER = '{ODBC Driver 18 for SQL Server}'

# Conexión con AAD Interactive
azure_conn_str = f"DRIVER={AZURE_DRIVER};SERVER={AZURE_SERVER};DATABASE={AZURE_DATABASE};Authentication=ActiveDirectoryInteractive"

# 📌 🔹 Conexión a **SQL Server LOCAL** (Autenticación de Windows)
LOCAL_SERVER = 'localhost\SQLEXPRESS'  # Puede ser 'localhost' o el nombre del servidor SQL
LOCAL_DATABASE = 'coches'  # Base de datos local
LOCAL_DRIVER = '{ODBC Driver 18 for SQL Server}'

# Conexión con autenticación de Windows
local_conn_str = f"DRIVER={LOCAL_DRIVER};SERVER={LOCAL_SERVER};DATABASE={LOCAL_DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes"

# Establecer las conexiones
conn_azure = pyodbc.connect(azure_conn_str)
conn_local = pyodbc.connect(local_conn_str)
cursor_local = conn_local.cursor()

# Mapeo de tipos de datos de Pandas a SQL Server
dtype_mapping = {
    'int64': 'BIGINT',  
    'float64': 'FLOAT',
    'object': 'NVARCHAR(MAX)',
    'bool': 'BIT',
    'datetime64[ns]': 'DATETIME2(7)'
}

# Nombres de las tablas
TABLE_NAME_stg_t = '[TFM1].[stg_t]'  # Agregar el esquema correcto
TABLE_NAME_web_lead_mod = '[TFM1].[web_lead_mod]'  # Agregar el esquema correcto
TABLE_NAME_web_path = '[TFM1].[web_path]'
TABLE_NAME_stg_l = '[TFM1].[stg_l]'  # Agregar el esquema correcto
TABLE_NAME_CC_mod = '[TFM1].[CC_mod]'  # Agregar el esquema correcto
TABLE_NAME_Coste_cd_mod = '[TFM1].[Coste_cd_mod]'  # Agregar el esquema correcto
TABLE_NAME_Mosaic = '[TFM1].[Mosaic]'  # Agregar el esquema correcto
TABLE_NAME_stg_p = '[TFM1].[stg_p]'  # Agregar el esquema correcto

# Leer los datos de Azure SQL en los DataFrames correspondientes
#df_stg_t = pd.read_sql(f"SELECT * FROM {TABLE_NAME_stg_t}", conn_azure)
#print(df_stg_t.dtypes)
#df_web_lead_mod = pd.read_sql(f"SELECT * FROM {TABLE_NAME_web_lead_mod}", conn_azure)
#print(df_web_lead_mod.dtypes)
#df_web_path = pd.read_sql(f"SELECT * FROM {TABLE_NAME_web_path}", conn_azure)
#print(df_web_path.dtypes)
#print(df_web_path.head())
#df_stg_l = pd.read_sql(f"SELECT * FROM {TABLE_NAME_stg_l}", conn_azure)
#print(df_stg_l.dtypes)
#print(df_stg_l.head())
#df_CC_mod = pd.read_sql(f"SELECT * FROM {TABLE_NAME_CC_mod}", conn_azure)
#print(df_CC_mod.dtypes)
df_Coste_cd_mod = pd.read_sql(f"SELECT * FROM {TABLE_NAME_Coste_cd_mod}", conn_azure)
# pasar la columna cpc a float
# Reemplazar '#¡DIV/0!' y otros posibles errores por 0
df_Coste_cd_mod['cpc'] = df_Coste_cd_mod['cpc'].replace(['#¡DIV/0!', '#DIV/0!', 'NaN', ''], 0)
df_Coste_cd_mod['cpc'] = df_Coste_cd_mod['cpc'].astype(float)
df_Coste_cd_mod = df_Coste_cd_mod.dropna()
print(df_Coste_cd_mod.dtypes)
print(df_Coste_cd_mod.head())
#df_Mosaic = pd.read_sql(f"SELECT * FROM {TABLE_NAME_Mosaic}", conn_azure)
#df_Mosaic.rename(columns={'Check': 'Check_flag'}, inplace=True)
#print(df_Mosaic.head())
#df_stg_p = pd.read_sql(f"SELECT * FROM {TABLE_NAME_stg_p}", conn_azure)
#print(df_stg_p.dtypes)
#print(df_stg_p.head())



# Función para crear la tabla e insertar datos
def create_and_insert_table(df, table_name):
    # Eliminar la tabla anterior si existe
    drop_table_sql = f"DROP TABLE IF EXISTS {table_name}"
    cursor_local.execute(drop_table_sql)
    conn_local.commit()

    if table_name == TABLE_NAME_web_path:
    ## Reemplazar valores NaN por None (NULL en SQL) en el DataFrame
        df = df.where(pd.notna(df), '')
    else:
        ## Reemplazar valores NaN por None (NULL en SQL) en el DataFrame
        df = df.where(pd.notna(df), None)
    

    # Generar la sentencia CREATE TABLE con los tipos correctos
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        {', '.join([f'{col} {dtype_mapping.get(str(dtype), "NVARCHAR(MAX)")}' for col, dtype in df.dtypes.items()])}
    )
    """
    print("Sentencia SQL generada:")
    print(create_table_sql)

    # Ejecutar la creación de la tabla
    cursor_local.execute(create_table_sql)
    conn_local.commit()

    # Preparar la sentencia INSERT con parámetros
    insert_sql = f"""
    INSERT INTO {table_name} ({', '.join(df.columns)}) 
    VALUES ({', '.join(['?' for _ in df.columns])})
    """

    # Insertar los datos utilizando parámetros
    for index, row in df.iterrows():
    
        values = tuple(None if pd.isna(val) else val for val in row)
        cursor_local.execute(insert_sql, values)
    
    conn_local.commit()

# Crear e insertar datos para cada DataFrame
#create_and_insert_table(df_stg_t, TABLE_NAME_stg_t)
#print("Tabla stg_t creada e insertada")
#create_and_insert_table(df_web_lead_mod, TABLE_NAME_web_lead_mod)
#print("Tabla web_lead_mod creada e insertada")
#create_and_insert_table(df_web_path, TABLE_NAME_web_path)
#print("Tabla web_path creada e insertada")
#create_and_insert_table(df_stg_l, TABLE_NAME_stg_l)
#print("Tabla stg_l creada e insertada")
#create_and_insert_table(df_CC_mod, TABLE_NAME_CC_mod)
#print("Tabla CC_mod creada e insertada")
create_and_insert_table(df_Coste_cd_mod, TABLE_NAME_Coste_cd_mod)
print("Tabla Coste_cd_mod creada e insertada")
#create_and_insert_table(df_Mosaic, TABLE_NAME_Mosaic)
#print("Tabla Mosaic creada e insertada")
#create_and_insert_table(df_stg_p, TABLE_NAME_stg_p)
#print("Tabla stg_p creada e insertada")

with open("C:\\TFG\\DW\\tabla trafico ML.sql", "r", encoding="utf-8") as f:
    sql_script = f.read()

# Ejecutar múltiples sentencias (split por ;)
for statement in sql_script.split(';'):
    if statement.strip():  # Evitar líneas vacías
        cursor_local.execute(statement)
conn_local.commit()

# Cerrar conexiones
cursor_local.close()
conn_local.close()
conn_azure.close()
