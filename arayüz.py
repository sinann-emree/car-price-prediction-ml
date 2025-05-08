import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score, mean_squared_error

# --- VERİYİ YÜKLE ---
df = pd.read_csv("updated_all_cars_with_brand.csv")
df["model"] = df["model"].astype(str).str.strip()
df["detected_brand"] = df["detected_brand"].astype(str).str.strip()
df["model_lower"] = df["model"].str.lower().str.strip()

# Gelişmiş model için gerekli sütunlar
features_needed = ["detected_brand", "model", "year", "mileage", "price",
                   "engineSize", "transmission", "fuelType", "tax", "mpg"]
df_model = df.dropna(subset=features_needed).copy()

# Label Encoding
le_brand = LabelEncoder().fit(df_model["detected_brand"])
le_model = LabelEncoder().fit(df_model["model"])
le_trans = LabelEncoder().fit(df_model["transmission"])
le_fuel = LabelEncoder().fit(df_model["fuelType"])

# Modeli yükle
reg_model = joblib.load("fiyat_modeli.pkl")

marka_model_sozluk = (
    df.groupby("detected_brand")["model"]
    .unique()
    .apply(lambda x: sorted([m for m in x if pd.notnull(m)]))
    .to_dict()
)

# --- TKINTER ARAYÜZ ---
root = tk.Tk()
root.title("\U0001F697 Gelişmiş Araç Fiyat Tahmin Sistemi")
root.geometry("750x700")
root.configure(bg="#f4f4f4")

style = ttk.Style()
style.configure("TLabel", font=("Segoe UI", 11), background="#f4f4f4")
style.configure("TButton", font=("Segoe UI", 10), padding=6)
style.configure("TCombobox", font=("Segoe UI", 10))

frame_inputs = ttk.LabelFrame(root, text="Araç Bilgileri", padding=(20, 10))
frame_inputs.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

tk.Label(frame_inputs, text="Marka").grid(row=0, column=0, padx=5, pady=5, sticky="w")
marka_var = tk.StringVar()
marka_combobox = ttk.Combobox(frame_inputs, textvariable=marka_var, width=30, state="readonly")
marka_combobox['values'] = sorted(marka_model_sozluk.keys())
marka_combobox.grid(row=0, column=1, padx=5, pady=5)

tk.Label(frame_inputs, text="Model").grid(row=1, column=0, padx=5, pady=5, sticky="w")
model_var = tk.StringVar()
model_combobox = ttk.Combobox(frame_inputs, textvariable=model_var, width=30, state="readonly")
model_combobox.grid(row=1, column=1, padx=5, pady=5)

def update_models(event):
    marka = marka_var.get()
    model_combobox['values'] = marka_model_sozluk.get(marka, [])
    model_var.set("")

marka_combobox.bind("<<ComboboxSelected>>", update_models)

tk.Label(frame_inputs, text="Yıl").grid(row=2, column=0, padx=5, pady=5, sticky="w")
yil_var = tk.StringVar()
tk.Entry(frame_inputs, textvariable=yil_var, width=33).grid(row=2, column=1, padx=5, pady=5)

tk.Label(frame_inputs, text="Kilometre").grid(row=3, column=0, padx=5, pady=5, sticky="w")
km_var = tk.StringVar()
tk.Entry(frame_inputs, textvariable=km_var, width=33).grid(row=3, column=1, padx=5, pady=5)

# --- TAHMİN BUTONU ---
tk.Button(frame_inputs, text="\U0001F50D Fiyat Tahmini Yap", command=lambda: benzer_araclari_bul()).grid(row=4, column=0, columnspan=2, pady=10)

# --- SONUÇ ALANI (SCROLLBAR İLE) ---
frame_sonuc = ttk.LabelFrame(root, text="Tahmini Fiyat & En Yakın Araçlar", padding=(20, 10))
frame_sonuc.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

scrollbar = tk.Scrollbar(frame_sonuc)
scrollbar.pack(side="right", fill="y")

sonuc_text = tk.Text(frame_sonuc, font=("Consolas", 10), bg="white", wrap="word", yscrollcommand=scrollbar.set)
sonuc_text.pack(fill="both", expand=True, padx=5, pady=5)
scrollbar.config(command=sonuc_text.yview)

# --- TAHMİN FONKSİYONU ---
def benzer_araclari_bul():
    sonuc_text.delete("1.0", tk.END)

    marka = marka_var.get()
    model = model_var.get()
    yil = yil_var.get()
    km = km_var.get()

    if not (marka and model and yil and km):
        sonuc_text.insert(tk.END, "❗ Lütfen tüm alanları doldurun.")
        return

    try:
        filtre = df[
            (df["detected_brand"].str.lower() == marka.lower()) &
            (df["model_lower"] == model.lower().strip())
        ].copy()

        if filtre.empty:
            sonuc_text.insert(tk.END, f"❗ {marka} {model} modeli veri setinde bulunamadı.")
            return

        yil = int(yil)
        km = int(km)

        filtre["fark"] = (filtre["year"] - yil).abs() + (filtre["mileage"] - km).abs() / 1000
        en_yakin = filtre.sort_values("fark").head(10)

        ref = en_yakin.iloc[0]

        veri = [[
            le_brand.transform([marka])[0],
            le_model.transform([model])[0],
            yil,
            km,
            ref["engineSize"],
            ref["tax"],
            ref["mpg"],
            le_trans.transform([ref["transmission"]])[0],
            le_fuel.transform([ref["fuelType"]])[0]
        ]]

        tahmin = reg_model.predict(veri)[0]

        sonuc = f"\U0001F697 Tahmini Fiyat: {int(tahmin):,} $\n\n\U0001F50D En Yakın 10 Araç:\n"
        for i, (_, row) in enumerate(en_yakin.iterrows(), 1):
            detay = (
                f"{i}. {row['detected_brand']} {row['model']} - "
                f"{row['year']} - {row['mileage']} km\n   "
                f"{row.get('transmission', 'N/A')} | {row.get('fuelType', 'N/A')} | "
                f"{row.get('engineSize', 'N/A')}L | Vergi: {row.get('tax', 'N/A')} | "
                f"Tüketim: {row.get('mpg', 'N/A')} mpg\n   "
                f"Fiyat: {int(row['price']):,} \n"
            )
            sonuc += detay + "\n"

        sonuc_text.insert(tk.END, sonuc)

    except Exception as e:
        sonuc_text.insert(tk.END, f"❌ Hata: {str(e)}")

root.mainloop()
