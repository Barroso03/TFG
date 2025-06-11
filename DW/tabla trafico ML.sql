----------------------------------------------------------------------------------------------------
-- TABLA LEAD (copia y transformaci�n de [TFM1].[stg_l] a la tabla local stg_l)
-- TABLA STG_T
----------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS stg_l;
-- 1. Crear una copia de la tabla original de leads y nombrarla como stg_l.
SELECT * 
INTO stg_l
FROM [TFM1].[stg_l];



-- 2. Eliminar filas duplicadas en stg_l, considerando una combinaci�n de m�ltiples columnas clave.
WITH CTE AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY identificador, 
                         tipo_de_lead, 
                         origen_del_lead, 
                         codigo_postal_cliente, 
                         permisos_contacto, 
                         dia_de_grabacion, 
                         hora_de_grabacion, 
                         codigo_estado, 
                         descripcion_estado, 
                         sub_estado, 
                         dia_de_cualificacion, 
                         hora_de_cualificacion, 
                         dia_subida_a_crm, 
                         codigo_concesionario_origen, 
                         Codigo_concesionario_final, 
                         estado_ultima_llamada, 
                         motivo_ultima_llamada, 
                         requiere_rellamada, 
                         fue_reactivado, 
                         estado_de_reactivacion, 
                         codigo_campana_origen_google, 
                         id_google_externo, 
                         correo_codificado, 
                         telefono_principal_codificado
            ORDER BY identificador
        ) AS fila
    FROM stg_l
)
DELETE FROM CTE 
WHERE fila > 1;

-- 3. Eliminar las filas en stg_l que no tienen coincidencia en la tabla web_lead_mod (por id_google_externo).
DELETE FROM stg_l
WHERE NOT EXISTS (
    SELECT 1 
    FROM [TFM1].[web_lead_mod] web
    WHERE stg_l.id_google_externo = web.id_google_externo
);

-- 4. Convertir la columna dia_de_grabacion al tipo DATE.
UPDATE stg_l  
SET dia_de_grabacion = CONVERT(DATE, dia_de_grabacion);

-- 5. Convertir la columna hora_de_grabacion al tipo TIME.
UPDATE stg_l
SET hora_de_grabacion = CONVERT(TIME, hora_de_grabacion);

-- 6. Agregar una nueva columna para combinar fecha y hora en formato DATETIME.
ALTER TABLE stg_l
ADD dia_hora_de_grabacion DATETIME;

-- 7. Combinar la fecha y la hora en la nueva columna dia_hora_de_grabacion.
UPDATE stg_l
SET dia_hora_de_grabacion = CAST(
    CONCAT(
        CONVERT(VARCHAR(10), dia_de_grabacion, 120), ' ', 
        CONVERT(VARCHAR(8), hora_de_grabacion, 108)
    ) AS DATETIME2(7)
);

-- 8. Eliminar las filas donde la columna descripcion_estado sea 'Duplicado'.
DELETE FROM stg_l
WHERE descripcion_estado = 'Duplicado';

-- 9. Eliminar duplicados por id_google_externo, dejando solo la fila con la fecha (dia_hora_de_grabacion) m�s reciente.
DELETE t
FROM stg_l t
JOIN (
    SELECT id_google_externo, MAX(dia_hora_de_grabacion) AS max_fecha
    FROM stg_l
    GROUP BY id_google_externo
) subquery ON t.id_google_externo = subquery.id_google_externo 
           AND t.dia_hora_de_grabacion < subquery.max_fecha;

-- 10. Eliminar duplicados de telefono_principal_codificado dejando solo la fila m�s reciente.
DELETE t
FROM stg_l t
JOIN (
    SELECT telefono_principal_codificado, MAX(dia_hora_de_grabacion) AS max_fecha
    FROM stg_l
    WHERE telefono_principal_codificado IS NOT NULL 
      AND telefono_principal_codificado != ''
    GROUP BY telefono_principal_codificado
) subquery 
ON t.telefono_principal_codificado = subquery.telefono_principal_codificado
   AND t.dia_hora_de_grabacion < subquery.max_fecha
WHERE t.telefono_principal_codificado IS NOT NULL 
  AND t.telefono_principal_codificado != '';

-- 11. Eliminar duplicados de correo_codificado dejando solo la fila m�s reciente.
DELETE t
FROM stg_l t
JOIN (
    SELECT correo_codificado, MAX(dia_hora_de_grabacion) AS max_fecha
    FROM stg_l
    WHERE correo_codificado IS NOT NULL 
      AND correo_codificado != ''
    GROUP BY correo_codificado
) subquery 
ON t.correo_codificado = subquery.correo_codificado
   AND t.dia_hora_de_grabacion < subquery.max_fecha
WHERE t.correo_codificado IS NOT NULL 
  AND t.correo_codificado != '';

-- 12. Eliminar los �ltimos 4 caracteres ('692A') de la columna telefono_principal_codificado, cuando corresponda.
UPDATE stg_l
SET telefono_principal_codificado = LEFT(telefono_principal_codificado, LEN(telefono_principal_codificado) - 4)
WHERE RIGHT(telefono_principal_codificado, 4) = '692A';

-- 13. Agregar una nueva columna 'tiene_trafico' de tipo INT a la tabla stg_l.
ALTER TABLE stg_l
ADD tiene_trafico INT;

DROP TABLE IF EXISTS stg_t;

-- 14. Crear una copia de la tabla original stg_t desde [TFM1].[stg_t].
SELECT * 
INTO stg_t
FROM [TFM1].[stg_t];

-- 15. En stg_t, eliminar duplicados basados en todas las columnas relevantes.
WITH CTE_T AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY email_codificado, telefono_codificado, campania_codificada, codigo_tienda, codigo_exposicion, 
                         codigo_exposicion2, codigo_producto, cod_oferta, contacto_cliente, 
                         codigo_postal, indicador_principal, indicador_trafico, estado_negociacion, 
                         fecha_oferta_dia, fecha_oferta_dia_hora, hora_visita, identificador_contacto, 
                         identificador_oferta, ubicacion, numero, total_ofertas, numero_orden, 
                         region, tipo_documento, tipo_tratamiento, origen_contacto, origen_negociacion, 
                         vendedor_codificado_final, PRODUCT_CODE_TRAFICO
            ORDER BY identificador_contacto
        ) AS fila
    FROM stg_t
)
DELETE FROM CTE_T 
WHERE fila > 1;

-- 16. Eliminar registros en stg_t donde fecha_oferta_dia_hora no se pueda convertir a DATETIME.
DELETE FROM stg_t
WHERE TRY_CONVERT(DATETIME, fecha_oferta_dia_hora, 103) IS NULL
  AND fecha_oferta_dia_hora IS NOT NULL;

-- 17. Actualizar la columna fecha_oferta_dia_hora en stg_t a DATETIME2(7) usando el formato 103.
UPDATE stg_t
SET fecha_oferta_dia_hora = CONVERT(DATETIME2(7), fecha_oferta_dia_hora, 103);

-- 18. Crear una tabla temporal (#temp_stg_t) con los registros filtrados para quedarnos solo con la fila m�s reciente
-- seg�n email o tel�fono.
SELECT *
INTO #temp_stg_t
FROM (
    SELECT *,
           COUNT(*) OVER (PARTITION BY COALESCE(email_codificado, telefono_codificado)) AS total_trafico,
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE(email_codificado, telefono_codificado)
               ORDER BY fecha_oferta_dia_hora DESC
           ) AS rn
    FROM stg_t
    WHERE email_codificado IS NOT NULL OR telefono_codificado IS NOT NULL
) AS base
WHERE rn = 1;

-- 19. Truncar (eliminar todos los registros de) la tabla original stg_t.
DROP TABLE stg_t;

-- 20. Insertar los registros filtrados desde la tabla temporal en stg_t.
SELECT * 
INTO stg_t
FROM #temp_stg_t;

-- 21. Eliminar la tabla temporal.
DROP TABLE #temp_stg_t;

-- 22. Agregar una columna id_trafico a stg_t con identidad (esto generar� un n�mero �nico para cada registro).
ALTER TABLE stg_t
ADD id_trafico INT IDENTITY(1,1) PRIMARY KEY;



-- 23. En stg_l, agregar la columna id_trafico para almacenar el identificador de tr�fico proveniente de stg_t.
ALTER TABLE stg_l
ADD id_trafico INT;

-- 24. Actualizar la columna tiene_trafico a 0 para las filas de stg_l que NO tienen coincidencia de correo o tel�fono en stg_t
-- (m�s adelante se actualizar� a 1 aquellas con coincidencia).
UPDATE l
SET tiene_trafico = 0
FROM stg_l l
WHERE NOT EXISTS (
    SELECT 1
    FROM stg_t t
    WHERE t.email_codificado = l.correo_codificado
       OR t.telefono_codificado = l.telefono_principal_codificado
);

-- 25. Actualizar la columna tiene_trafico a 1 para las filas que tengan coincidencia (aquellas que a�n tengan NULL se ponen a 1).
UPDATE l
SET tiene_trafico = 1
FROM stg_l l
WHERE tiene_trafico IS NULL;


-- 26. Actualizar stg_l asignando id_trafico de stg_t cuando coincida el correo.
UPDATE l2
SET l2.id_trafico = t2.id_trafico
FROM stg_l l2
INNER JOIN stg_t t2
    ON l2.correo_codificado = t2.email_codificado;

-- 27. Para las filas de stg_l que a�n tienen id_trafico NULL, actualizar usando el tel�fono.
UPDATE l2
SET l2.id_trafico = t2.id_trafico
FROM stg_l l2
INNER JOIN stg_t t2
    ON l2.telefono_principal_codificado = t2.telefono_codificado
WHERE l2.id_trafico IS NULL;

-- 28. Fk entre stg_l y stg_t
ALTER TABLE stg_l
ADD CONSTRAINT FK_stg_l_stg_t_id_trafico
FOREIGN KEY (id_trafico) REFERENCES stg_t(id_trafico);



----------------------------------------------------------------------------------------------------
-- TABLA WEB_LEAD_MOD
-- (Copia la tabla original y se realizan actualizaciones y relaciones)
----------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS web_lead_mod;
-- 1. Crear una copia de la tabla original web_lead_mod y llamarla web_lead_mod.
SELECT *  
INTO web_lead_mod  
FROM [TFM1].[web_lead_mod];

-- 2. Agregar la columna id_web a web_lead_mod, de tipo INT con identidad, y definirla como clave primaria.
ALTER TABLE web_lead_mod  
ADD id_web INT IDENTITY(1,1) PRIMARY KEY;

-- 3. Agregar una columna id_web en stg_l para almacenar el valor de id_web de web_lead_mod.
ALTER TABLE stg_l  
ADD id_web INT;

-- 4. Actualizar stg_l asignando el id_web de web_lead_mod (usando la coincidencia de id_google_externo).
UPDATE l  
SET l.id_web = w.id_web  
FROM stg_l l  
INNER JOIN web_lead_mod w 
    ON w.id_google_externo = l.id_google_externo;

-- 5. Agregar una restricci�n de clave for�nea en stg_l que haga referencia a web_lead_mod (id_web).
ALTER TABLE stg_l  
ADD CONSTRAINT fk_web_lead FOREIGN KEY (id_web)  
REFERENCES web_lead_mod (id_web);


----------------------------------------------------------------------------------------------------
-- TABLA WEB_PATH
-- (Copia, conversi�n de columnas y relaci�n con web_lead_mod)
----------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS web_path;
-- 1. Crear una copia de la tabla original web_path y llamarla web_path.
SELECT * 
INTO web_path
FROM [TFM1].[web_path];

-- 2. Convertir las columnas num�ricas a INT en web_path.
ALTER TABLE web_path ALTER COLUMN OFFERS INT;
ALTER TABLE web_path ALTER COLUMN HOME INT;
ALTER TABLE web_path ALTER COLUMN SHOWROOM INT;
ALTER TABLE web_path ALTER COLUMN PROMOTIONS INT;
ALTER TABLE web_path ALTER COLUMN OTHERS INT;
ALTER TABLE web_path ALTER COLUMN CC INT;
ALTER TABLE web_path ALTER COLUMN DLR INT;
ALTER TABLE web_path ALTER COLUMN FLEET INT;
ALTER TABLE web_path ALTER COLUMN PV INT;

-- 3. Agregar la columna id_web_path a web_path, con identidad y clave primaria.
ALTER TABLE web_path
ADD id_web_path INT IDENTITY(1,1) PRIMARY KEY;

-- 4. Agregar la columna id_web_path en web_lead_mod para almacenar el valor de web_path.
ALTER TABLE web_lead_mod
ADD id_web_path INT;

-- 5. Actualizar web_lead_mod asignando id_web_path desde web_path usando la columna user_id.
UPDATE wlm
SET wlm.id_web_path = wp.id_web_path
FROM web_lead_mod wlm
INNER JOIN web_path wp 
    ON wlm.user_id = wp.user_id;

-- 6. Agregar una restricci�n de clave for�nea en web_lead_mod para la columna id_web_path.
ALTER TABLE web_lead_mod
ADD CONSTRAINT FK_web_lead_mod_web_path
FOREIGN KEY (id_web_path) REFERENCES web_path(id_web_path);


----------------------------------------------------------------------------------------------------
-- TABLA COSTE_CD_MOD
-- (Agrupa datos y establece relaciones)
----------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS Coste_cd_mod;
-- 1. Crear una nueva tabla Coste_cd_mod copiando y agrupando datos de la tabla original, dejando solo campa�as con m�s de un registro.
SELECT campaigncod,
       AVG(ctr) AS avg_ctr,
       AVG(cpc) AS avg_cpc,
       AVG(cpi) AS avg_cpi
INTO Coste_cd_mod
FROM [TFM1].[Coste_cd_mod]
GROUP BY campaigncod
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC;

-- 2. Agregar una columna id_Coste_cd_mod autoincremental (clave primaria) a Coste_cd_mod.
ALTER TABLE Coste_cd_mod
ADD id_Coste_cd_mod INT IDENTITY(1,1) PRIMARY KEY;

-- 3. Agregar la columna id_Coste_cd_mod en web_lead_mod para establecer la relaci�n.
ALTER TABLE web_lead_mod
ADD id_Coste_cd_mod INT NULL;

-- 4. Actualizar web_lead_mod asignando id_Coste_cd_mod bas�ndose en la coincidencia entre campaign_cod y campaigncod.
UPDATE w
SET w.id_Coste_cd_mod = c.id_Coste_cd_mod
FROM web_lead_mod w
INNER JOIN Coste_cd_mod c
    ON w.campaign_cod = c.campaigncod;

-- 5. Agregar una clave for�nea en web_lead_mod para la columna id_Coste_cd_mod.
ALTER TABLE web_lead_mod
ADD CONSTRAINT FK_web_lead_Coste_cd_mod
FOREIGN KEY (id_Coste_cd_mod)
REFERENCES Coste_cd_mod(id_Coste_cd_mod);


----------------------------------------------------------------------------------------------------
-- TABLA MOSAIC
-- (Copia, formatea CP y establece relaciones)
----------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS Mosaic;
-- 1. Crear una copia de la tabla original Mosaic y llamarla Mosaic.
SELECT * 
INTO Mosaic
FROM [TFM1].[Mosaic];

-- 2. Actualizar la columna CP para que los c�digos postales de 4 d�gitos tengan un '0' delante.
UPDATE Mosaic
SET CP = '0' + CP
WHERE LEN(CP) = 4;

-- 3. Agregar la columna id_mosaic autoincremental como clave primaria en Mosaic.
ALTER TABLE Mosaic
ADD id_mosaic INT IDENTITY(1,1) PRIMARY KEY;

-- 4. Agregar la columna id_mosaic en stg_l para relacionarla con Mosaic.
ALTER TABLE stg_l
ADD id_mosaic INT;

-- 5. Actualizar stg_l para agregar un '0' delante de codigo_postal_cliente cuando tenga 4 d�gitos.
UPDATE stg_l
SET codigo_postal_cliente = '0' + codigo_postal_cliente
WHERE LEN(codigo_postal_cliente) = 4;

-- 6. Asignar el id_mosaic correspondiente a cada lead en stg_l uniendo por codigo_postal_cliente y CP.
UPDATE s
SET s.id_mosaic = m.id_mosaic
FROM stg_l s
INNER JOIN Mosaic m 
    ON s.codigo_postal_cliente = m.CP;

-- 7. Crear una clave for�nea en stg_l para la columna id_mosaic.
ALTER TABLE stg_l
ADD CONSTRAINT FK_mosaic_lead 
FOREIGN KEY (id_mosaic)
REFERENCES Mosaic(id_mosaic);


----------------------------------------------------------------------------------------------------
-- TABLA CC_MOD
-- (Copia y establece relaci�n con web_lead_mod)
----------------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS CC_mod;
-- 1. Crear una copia de la tabla original CC_mod y llamarla CC_mod.
SELECT * 
INTO CC_mod
FROM [TFM1].[CC_mod];

-- 2. Agregar la columna id_CC_mod en web_lead_mod para establecer la relaci�n.
ALTER TABLE web_lead_mod
ADD id_CC_mod INT;

-- 3. Agregar la columna id_CC_mod autoincremental como clave primaria en CC_mod.
ALTER TABLE CC_mod 
ADD id_CC_mod INT IDENTITY(1,1) PRIMARY KEY;

-- 4. Actualizar web_lead_mod asignando id_CC_mod uniendo con CC_mod seg�n id_oferta_mod e id_google_externo.
UPDATE l
SET l.id_CC_mod = c.id_CC_mod
FROM web_lead_mod l
INNER JOIN CC_mod c 
    ON c.id_oferta_mod = l.id_google_externo;

-- 5. Agregar una clave for�nea en web_lead_mod que haga referencia a id_CC_mod de CC_mod.
ALTER TABLE web_lead_mod
ADD CONSTRAINT FK_CC_mod_lead 
FOREIGN KEY (id_CC_mod)
REFERENCES CC_mod(id_CC_mod);






----------------------------------------------------------------------------------------------------
-- TABLA TRAFICO
DROP TABLE IF EXISTS stg_p;
-- Paso 1: Crear la tabla stg_p a partir de [TFM1].[stg_p]
SELECT * 
INTO stg_p
FROM [TFM1].[stg_p];

-- Paso 2: Eliminar duplicados en stg_p dejando solo una fila por grupo
WITH CTE AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY 
                   contacto_cliente,
                   numero_cliente,
                   identificador_contacto,
                   codigo_tienda,
                   estado_actual,
                   codigo_postal,
                   estimado,
                   tipo_tratamiento,
                   codigo_exposicion,
                   total_pedidos,
                   numero_auxiliar,
                   oferta_codigo,
                   identificador_matriculacion,
                   fecha_del_pedido,
                   fecha_de_matriculacion,
                   telefono_codificado,
                   correo_codificado,
                   codigo_exposicion_alternativo,
                   dia_del_pedido,
                   dia_de_matriculacion,
                   tipo_documento,
                   codigo_tipo_venta,
                   ubicacion,
                   region,
                   bastidor_codificado_final,
                   codigo_comercial_codificado,
                   PRODUCT_CODE_TRAFICO
               ORDER BY PRODUCT_CODE_TRAFICO
           ) AS fila
    FROM stg_p
)
DELETE FROM CTE 
WHERE fila > 1;

-- Paso 3: Agregar la columna ha_comprado a stg_t
ALTER TABLE stg_t
ADD ha_comprado INT;

-- Paso 4: Marcar con 1 si hay coincidencia por correo
UPDATE stg_t
SET ha_comprado = 1
FROM stg_t t
JOIN stg_p v ON t.email_codificado = v.correo_codificado;

-- Paso 5: Marcar con 1 si hay coincidencia por tel�fono (solo si no ten�a ya un 1)
UPDATE stg_t
SET ha_comprado = 1
FROM stg_t t
JOIN stg_p v ON t.telefono_codificado = v.telefono_codificado
WHERE ha_comprado IS NULL;

-- Paso 6: Marcar con 0 si no hay coincidencia ni por correo ni por tel�fono
UPDATE stg_t
SET ha_comprado = 0
WHERE ha_comprado IS NULL;



