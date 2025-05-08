import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


# Verisetlerini oku
dataAudi = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/audi.csv")
dataBmw = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/bmw.csv")
dataCclass = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/cclass.csv")
dataFocus = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/focus.csv")
dataFord = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/ford.csv")
dataHyundi = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/hyundi.csv")
dataMerc = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/merc.csv")
dataOpel = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/opel.csv")
dataSkoda = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/skoda.csv")
dataToyota = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/toyota.csv")
dataVw = pd.read_csv("C:/Users/sinan/Desktop/fiyat tahmin/datasets/vw.csv")

# Verisetlerinin kolonlarını kontrol et
print(dataAudi.columns)
print(dataBmw.columns)
print(dataCclass.columns) # tax yok
print(dataFocus.columns) # tax yok
print(dataFord.columns)
print(dataHyundi.columns) # tax adı farklı (değiştir)
print(dataMerc.columns)
print(dataOpel.columns)
print(dataSkoda.columns)
print(dataToyota.columns)
print(dataVw.columns)

# Verileri temizle ve düzenle
dataframes = [
    dataAudi, dataBmw, dataCclass, dataFocus, dataFord,
    dataHyundi, dataMerc, dataOpel, dataSkoda, dataToyota, dataVw
]

target_columns = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize', 'source']
cleaned_dataframes = []

for df in dataframes:
    if 'tax(£)' in df.columns:
        df = df.rename(columns={'tax(£)': 'tax'})
    for col in target_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[target_columns]
    cleaned_dataframes.append(df)

birlesik_df = pd.concat(cleaned_dataframes, ignore_index=True)
all_data = birlesik_df

# 'tax' ve 'mpg' sütunlarındaki eksik verileri her marka için ortalama ile doldur
all_data['tax'] = all_data.groupby('model')['tax'].transform(lambda x: x.fillna(x.mean()))
all_data['mpg'] = all_data.groupby('model')['mpg'].transform(lambda x: x.fillna(x.mean()))

# Kategorik sütunları sayısallaştır
label_cols = ['fuelType', 'transmission']
le = LabelEncoder()
for col in label_cols:
    all_data[col] = le.fit_transform(all_data[col])

# Engine size 0 olanları listeleyelim
engine_size_zero = all_data[all_data['engineSize'] == 0]
print(engine_size_zero)

# 'model' adı 'i3' ve '230' olanları silelim
all_data_cleaned = all_data[~all_data['model'].isin(['i3', '230'])]
all_data = all_data_cleaned

# 'source' sütununu kaldır
all_data.drop(columns="source", inplace=True)

# model sütununu sayısallaştır
le_model = LabelEncoder()
all_data_cleaned['model_num'] = le_model.fit_transform(all_data_cleaned['model'])
all_data = all_data_cleaned

# IQR ile aykırı değerleri silme
Q1 = all_data[['year', 'mileage']].quantile(0.25)
Q3 = all_data[['year', 'mileage']].quantile(0.75)
IQR = Q3 - Q1

# Aykırı değerleri çıkar
all_data = all_data[~((all_data[['year', 'mileage']] < (Q1 - 1.5 * IQR)) | (all_data[['year', 'mileage']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Verileri normalleştir
scaler = MinMaxScaler()
all_data[['year', 'mileage', 'price', 'tax', 'mpg', 'engineSize', 'model_num', 'transmission', 'fuelType']] = scaler.fit_transform(
    all_data[['year', 'mileage', 'price', 'tax', 'mpg', 'engineSize', 'model_num', 'transmission', 'fuelType']]
)

# Sayısal sütunları seçelim
numerical_cols = all_data.select_dtypes(include=['float64', 'int64']).columns
numerical_data = all_data[numerical_cols]

# Korelasyon matrisini hesaplayalım
correlation_matrix = numerical_data.corr()
price_correlation = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlation)

# Verilerin eksik olup olmadığını kontrol edelim
print(all_data.isnull().sum())

# Özellikler (X) ve hedef (y) belirle
X = all_data_cleaned.drop(columns=['price', 'model'])  # model ve brand metin, price hedef
y = all_data_cleaned['price']

# Eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verileri standartlaştır
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Modeli
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression:")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

# K-Nearest Neighbors Modeli
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\nK-Nearest Neighbors:")
print("R2 Score:", r2_score(y_test, y_pred_knn))
print("MSE:", mean_squared_error(y_test, y_pred_knn))

# Sonuçları bir DataFrame olarak göstermek
results = {
    'Model': ['Linear Regression', 'K-Nearest Neighbors'],
    'R2 Score': [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_knn)],
    'MSE': [mean_squared_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_knn)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_knn))]
}

accuracy_df = pd.DataFrame(results)
print(accuracy_df)
