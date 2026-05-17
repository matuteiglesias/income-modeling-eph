
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

#Grafico de suma de Ingresos por asistencia educativa
data.groupby('P08')['P47T'].sum().plot(kind='bar', figsize=(10, 6), title='Total Ingresos por asistencia educativa')
plt.show()

#Grafico de suma de Ingresos por sexo
data.groupby('P02')['P47T'].sum().plot(kind='bar', figsize=(10, 6), title='Total Ingresos por sexo')
plt.show()

#Grafico de suma de Ingresos por sexo
data.groupby('P10')['P47T'].sum().plot(kind='bar', figsize=(10, 6), title='Total Ingresos por completitud de educacion')
plt.show()

print(data.groupby('ANO4')['P47T'].describe())
print(data[data['P47T']>0].groupby('ANO4')['P47T'].describe())

data['logP47T'] = np.where(data['P47T'] > 0, np.log10(data['P47T']), np.nan)
print(data.groupby('ANO4')['P47T'].describe())
print(data.groupby('ANO4')['logP47T'].describe())


fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 fila, 2 columnas

# Primer boxplot
data[(data['P47T'] < 1000000) & (data['P47T'] > 0)].boxplot(
    column='P47T', by='ANO4', grid=False, ax=axes[0]
)
axes[0].set_title("Boxplot of P47T by ANO4")

# Segundo boxplot
data[data['logP47T'] > 0].boxplot(
    column='logP47T', by='ANO4', grid=False, ax=axes[1]
)
axes[1].set_title("Boxplot of logP47T by ANO4")

plt.suptitle("Comparación de Boxplots por ANO4")  # título general
plt.tight_layout()
plt.show()


data_viz = data[(data['P47T'] < 50000) & (data['P47T'] > 0)]
plt.figure(figsize=(10, 6))
plt.title("Histogram of P47T (Filtered)")
data_viz['P47T'].hist(bins=100)
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Histogram of logP47T")
data['logP47T'].hist(bins=100)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Primer gráfico: por Región
region_sum = data.groupby('Region')['P47T'].sum().reset_index()
region_sum = region_sum.sort_values(by='P47T', ascending=False)
region_sum.plot(kind='bar', x='Region', y='P47T', legend=False, ax=axes[0])
axes[0].set_title("Total Ingresos por Región")

# Segundo gráfico: por Aglomerado
alg_sum = data.groupby('AGLOMERADO')['P47T'].sum().reset_index()
alg_sum = alg_sum.sort_values(by='P47T', ascending=False)
alg_sum.plot(kind='bar', x='AGLOMERADO', y='P47T', legend=False, ax=axes[1])
axes[1].set_title("Total Ingresos por Aglomerado")

# Ajustar diseño y mostrar
plt.suptitle("Comparación de Ingresos por Región y Aglomerado")
plt.tight_layout()
plt.show()

#Identificar Outliers de P47T
Q1 = data['P47T'].quantile(0.25)
Q3 = data['P47T'].quantile(0.75)
logQ1 = data['logP47T'].quantile(0.25)
logQ3 = data['logP47T'].quantile(0.75)
IQR = Q3 - Q1
logIQR = logQ3 - logQ1
outliers = data[(data['P47T'] < (Q1 - 1.5 * IQR)) | (data['P47T'] > (Q3 + 1.5 * IQR))]
print(f'Number of outliers in P47T: {outliers.shape[0]}')
log_outliers = data[(data['logP47T'] < (logQ1 - 1.5 * logIQR)) | (data['logP47T'] > (logQ3 + 1.5 * logIQR))]
print(f'Number of outliers in P47T: {log_outliers.shape[0]}')

#correlation matrix for numerical features
numerical_features = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_features.corr()
plt.figure(figsize=(25, 25))
plt.title("Correlation Matrix")
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.show()