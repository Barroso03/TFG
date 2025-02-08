from sqlalchemy import create_engine, text
import pandas as pd

# Conectar a la base de datos de origen (coches) y destino (DwCoches)
db_source = "mysql+pymysql://root:Mimadre1974@localhost:3306/coches"
db_dw = "mysql+pymysql://root:Mimadre1974@localhost:3306/DwCoches"

engine_source = create_engine(db_source)
engine_dw = create_engine(db_dw)

# Crear la base de datos del Data Warehouse si no existe
connection = engine_source.connect()
connection.execute(text("CREATE DATABASE IF NOT EXISTS DwCoches"))
connection.close()
print("Base de datos 'DwCoches' verificada o creada exitosamente.")

# Obtener todas las tablas de la base de datos de origen
connection_source = engine_source.connect()
tables_result = connection_source.execute(text("SHOW TABLES"))
tables = [row[0] for row in tables_result]
connection_source.close()

print(f"Tablas encontradas en 'coches': {tables}")

# Definir un diccionario para cambiar los nombres de las tablas
table_name_mapping = {
    "lkp_pord0_mysql": "dbo.lkp_producto",  
    "lkp_tv_mysql": "dbo.lkp_ventas",
    "stg_prod_mysql": "dbo.producto",
    "stg_l_mysql": "dbo.lead",
    "stg_t_mysql": "dbo.trafico",
    "stg_p_mysql": "dbo.pedido",
    "stg_sale_mysql": "dbo.venta"
}

# Obtener los bastidores en ambas tablas
query_bastidores = """
SELECT p.bastidor_codificado_final
FROM stg_p_mysql p
INNER JOIN stg_sale_mysql s ON s.codigo_bas_codificado = p.bastidor_codificado_final
"""

bastidores_comunes = pd.read_sql(query_bastidores, engine_source)
bastidores_comunes = bastidores_comunes['bastidor_codificado_final'].tolist()

# Importar todas las tablas a 'DwCoches' con los nuevos nombres
for table in tables:
    new_table_name = table_name_mapping.get(table, table)  # Si no está en el diccionario, mantiene el mismo nombre
    print(f"Importando tabla '{table}' como '{new_table_name}'...")

    # Leer los datos de la tabla
    df = pd.read_sql(f"SELECT * FROM {table}", engine_source)
    
    # Si la tabla es 'stg_p_mysql', filtramos
    if table == "stg_p_mysql":
        df = df[df['bastidor_codificado_final'].isin(bastidores_comunes)]
        
        # Eliminar filas duplicadas
        df = df.drop_duplicates()
        print(f"Filas duplicadas eliminadas en '{table}'.")
        
        
        # Convertir fechas
        df['fecha_del_pedido'] = pd.to_datetime(df['fecha_del_pedido'], format='%d/%m/%Y %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%d')
        df['fecha_de_matriculacion'] = pd.to_datetime(df['fecha_de_matriculacion'], format='%d/%m/%Y %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%d')
        df = df[df['fecha_de_matriculacion'] >= '2021-01-02']
    
    # Si la tabla es 'stg_sale_mysql', filtramos
    if table == "stg_sale_mysql":
        df = df[df['codigo_bas_codificado'].isin(bastidores_comunes)]
        
        # Convertir fechas
        df['fecha_matricula_formateada'] = pd.to_datetime(df['fecha_matricula_formateada'], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y-%m-%d')
        df['fecha_aviso'] = pd.to_datetime(df['fecha_aviso'], format='%d/%m/%Y %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%d')
        df['fecha_alta_matricula'] = pd.to_datetime(df['fecha_alta_matricula'], format='%d/%m/%Y %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%d')
        
    
    
    # Guardar los datos en la base de datos del Data Warehouse con el nuevo nombre
    df.to_sql(new_table_name, con=engine_dw, if_exists="replace", index=False)

    print(f"Tabla '{table}' importada correctamente como '{new_table_name}' en 'DwCoches'.")

print("Proceso ETL finalizado exitosamente.")






