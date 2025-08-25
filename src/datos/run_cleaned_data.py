# src/data/processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ____________ 0_Establecer configuración de logging ____________
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('procesador-datos')

# ____________ 1_Cargar los datos ____________
def load_data(file_path):
    """Cargar datos desde un archivo CSV."""
    logger.info(f"Cargando datos desde {file_path}")
    return pd.read_csv(file_path) # Cargar el archivo CSV

# ____________ 2_Limpiar los datos ____________
def clean_data(df):
    """Limpiar el conjunto de datos manejando valores faltantes y atípicos."""
    logger.info("Limpiando el conjunto de datos")

    # Hacer una copia para evitar modificar el dataframe original
    df_cleaned = df.copy()

    # Manejar valores faltantes
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()

        if missing_count > 0:
            logger.info(f"Se encontraron {missing_count} valores faltantes en {column}")

            # Para columnas numéricas, llenar con la mediana
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column] = df_cleaned[column].fillna(median_value)
                logger.info(f"Se llenaron los valores faltantes en {column} con la mediana: {median_value}")

            # Para columnas categóricas, llenar con la moda
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                logger.info(f"Se llenaron los valores faltantes en {column} con la moda: {mode_value}")

    # Manejar valores atípicos en el precio (variable objetivo)
    # Usando el método IQR para identificar valores atípicos
    Q1 = df_cleaned['price'].quantile(0.25)
    Q3 = df_cleaned['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar valores atípicos extremos
    outliers = df_cleaned[(df_cleaned['price'] < lower_bound) |
                        (df_cleaned['price'] > upper_bound)]

    if not outliers.empty:
        logger.info(f"Se encontraron {len(outliers)} valores atípicos en la columna de precios")
        df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) &
                                (df_cleaned['price'] <= upper_bound)]
        logger.info(f"Se eliminaron los valores atípicos. Nueva forma del conjunto de datos: {df_cleaned.shape}")

    return df_cleaned

# ____________ 3_Procesar los datos ____________
def process_data(input_file, output_file):
    """Definir la tubería completa de procesamiento de datos."""

    # Crear el directorio de salida si no existe
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    df = load_data(input_file)
    logger.info(f"Se cargaron los datos con forma: {df.shape}")

    # Limpiar datos
    df_cleaned = clean_data(df)

    # Guardar datos procesados
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Se guardaron los datos procesados en {output_file}")

    return df_cleaned

# ____________ 4_Ejecutar procesamiento ____________
if __name__ == "__main__":
    # Ejemplo de uso
    process_data(
        input_file="datos/crudo/house_data.csv",
        output_file="datos/procesado/cleaned_house_data.csv"
    )
