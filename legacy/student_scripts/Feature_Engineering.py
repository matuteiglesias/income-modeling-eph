from pathlib import Path
import pandas as pd
import numpy as np

# Carpeta base = ubicación del script actual
base_dir = Path(__file__).resolve().parent
file_path22 = base_dir / "../../data/training/EPHARG_train_22.csv"
file_path23 = base_dir / "../../data/training/EPHARG_train_23.csv"
file_path24 = base_dir / "../../data/training/EPHARG_train_24.csv"
file_path25 = base_dir / "../../data/training/EPHARG_train_25.csv"

data22 = pd.read_csv(file_path22)
data23 = pd.read_csv(file_path23)
data24 = pd.read_csv(file_path24)
data24.drop(columns = ["V2_01_M", "V2_02_M", "V2_03_M","V5_01_M", "V5_02_M", "V5_03_M"], inplace=True)
data25 = pd.read_csv(file_path25)
data25.drop(columns = ["V2_01_M", "V2_02_M", "V2_03_M","V5_01_M", "V5_02_M", "V5_03_M"], inplace=True)
data = pd.concat([data22, data23, data24, data25], ignore_index=True)
pd.set_option('display.max_columns', None)

data['logP47T'] = np.where((data['P47T'] > 0), np.log10(data['P47T']), np.nan)
data_original = data.copy()

#Variables categoricas
data['V01'] = data['V01'].replace({0:9})
data['H05'] = data['H05'].replace({0:9}) 
data['H06'] = data['H06'].replace({0:9}) 
data['H07'] = data['H07'].replace({0:1}) #Al ser tan pocos los nulos, los agrupo con la clase mayoritaria
data['H08'] = data['H08'].replace({0:1}) #Al ser tan pocos los nulos, los agrupo con la clase mayoritaria
data['H09'] = data['H09'].replace({0:1}) #Al ser tan pocos los nulos, los agrupo con la clase mayoritaria
data['H10'] = data['H10'].replace({0:1}) #Al ser tan pocos los nulos, los agrupo con la clase mayoritaria
data['H11'] = data['H11'].replace({0:9}) 
data['H12'] = data['H12'].replace({0:9})
data['H13'] = data['H13'].replace({0:9})
data['H14'] = data['H14'].replace({0:9})
data['P05'] = data['P05'].replace({0:1}) #Al ser tan pocos los nulos, los agrupo con la clase mayoritaria
data['P07'] = data['P07'].replace({0:1}) #Al ser tan pocos los nulos, los agrupo con la clase mayoritaria
data['P08'] = data['P08'].replace({0:9}) 
data['P09'] = data['P09'].astype("string").replace({"0":"Z_NO_APLICA"}) 
data['P10'] = data['P10'].replace({0:9})
data['PP07G_59'] = data['PP07G_59'].replace({1:0})  #Acá 0 no es nulo! 5: No tiene ninguno, 0: tiene al menos 1 de ellos.
data['PROP'] = data['PROP'].replace({0:9})
data["CONDACT"] = data["CONDACT"].replace({0:9})

#Dropeo las variables pp07g_1,...,4; pp07h, pp07i, pp07j, pp07k ya que pp07g_59 me da una vision mas resumida de la calidad del trabajo de la persona. Resume informalidad, y 
data.drop(columns = ["PP07G1", "PP07G2", "PP07G3","PP07G4","PP07H","PP07I","PP07J","PP07K"], inplace=True)

#Dropeo las variables p07 y p08 ya que p09 y p10 me da una vision mas detallada de la calidad educativa de la persona.
data.drop(columns = ["P07","P08"], inplace=True)

tiene_banio = (data["H10"] == 1)

banio_mejorado = (
    tiene_banio &
    data["H12"].isin([1, 2])
)

# 3) Indice de saneamiento. Dejo esta variable como numerica ordinal en vez de categorica.
data["sanitacion_nivel"] = np.select(
    condlist=[
        ~tiene_banio,     # Nivel 0: no tiene baño
        banio_mejorado,   # Nivel 2: baño mejorado
        tiene_banio       # Nivel 1: baño precario
    ],
    choicelist=[
        0,
        2,
        1
    ],
    default=9
)

#Uso indice saneamiento como indice para cualidades del  baño. Por esto elimino H10,H11,H12
data.drop(columns=["H10","H11","H12"], inplace=True)

#Todas las columnas de arriba son categoricas, las convierto a category
data = data.astype({'ANO4': 'category', 'TRIMESTRE': 'category', 'AGLOMERADO': 'category', 'V01': 'category', 'H05': 'category', 
                    'H06': 'category', 'H07': 'category', 'H08': 'category', 'H09':'category', 'PROP':'category', 
                    'H14':'category','H13':'category','CAT_INAC':'category', 'CAT_OCUP':'category', 'CH07':'category', 'P10':'category','P05':'category', 
                    'Region':'category', 'P09':'category','CONDACT':'category','PP07G_59':'category', 'P02':'category'})

categorical_columns = data.select_dtypes(include=['category']).columns

target_cols_reg = { 'P21', 'T_VI', 'V12_M', 'V2_M', 'V3_M', 'V5_M', 'TOT_P12', 'PP08D1'}

for col in target_cols_reg:
    new_col_name = 'log' + col
    data[new_col_name] = np.where((data[col] > 0), np.log10(data[col]), np.nan)


def get_age_group(age):
    if age <= 3:
        return '0-3'
    elif age <=8:
        return '4-8'
    elif age <= 12:
        return '9-12'
    elif age <= 17:
        return '13-17'
    elif age <= 24:
        return '18-24'
    elif age <= 64:
        return '25-64'
    else:
        return '65+'

data['Rango_Etario'] = data['P03'].apply(get_age_group)

age_group_counts = data.groupby(['CODUSU', 'Rango_Etario']).size().unstack(fill_value=0)

for age_group in age_group_counts.columns:
    data[f'Personas_{age_group}'] = data['CODUSU'].map(age_group_counts[age_group])

data.drop(columns=['Rango_Etario'], inplace=True)

#Maximo nivel educativo en el hogar
import numpy as np
import pandas as pd

p09_num = pd.to_numeric(
    data['P09'].replace("Z_NO_APLICA", np.nan),
    errors="coerce"
)

data['Max_Nivel_Educativo'] = p09_num.groupby(data['CODUSU']).transform('max').fillna(0)

