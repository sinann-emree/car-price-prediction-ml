# Car Price Prediction ML 🚗💰

Bu proje, marka, model, yıl, kilometre ve yakıt türü gibi çeşitli araç özelliklerini kullanarak 2. el otomobil fiyatlarını makine öğrenmesi teknikleriyle tahmin eden bir uygulamadır. 

## 📌 Proje Hakkında
Bu depo; veri ön işleme, özellik mühendisliği (feature engineering), makine öğrenmesi modelinin eğitilmesi ve değerlendirilmesi gibi veri biliminin temel süreçlerini barındırmaktadır. Geliştirilen model sayesinde, kullanıcılar istedikleri aracın özelliklerini sisteme girerek yaklaşık bir piyasa değeri tahmini alabilirler.

## 🚀 Öne Çıkan Özellikler
* **Kapsamlı Veri Seti:** Audi, BMW, Ford, Hyundai, Mercedes, Opel, Skoda, Toyota ve VW gibi birçok farklı markaya ait veri setleri birleştirilmiş ve temizlenerek modele sunulmuştur.
* **Önceden Eğitilmiş Model:** Proje sonucunda elde edilen en başarılı model `final_model.pkl` olarak kaydedilmiş olup, her seferinde yeniden eğitim gerektirmeden hızlıca kullanılabilir.
* **Kullanıcı Arayüzü:** `arayuz.py` dosyası sayesinde teknik olmayan kullanıcılar da sisteme araç özelliklerini girip anında fiyat tahmini sonucu alabilirler.

## 📁 Dosya Yapısı
* `aiproject (1).py`: Verilerin işlendiği, modelin eğitildiği ve performans testlerinin yapıldığı ana kaynak kod dosyası.
* `arayuz.py`: Fiyat tahmini işlemlerini kullanıcı dostu bir ekranda sunan grafiksel arayüz (GUI).
* `final_model.pkl`: Modelin dışa aktarılmış, kullanıma hazır ağırlık dosyası.
* `*.csv`: Markalara ait ham veri dosyaları ve veri ön işleme sonrasında elde edilen birleştirilmiş ana veri setleri (örneğin: `updated_all_cars_with_brand.csv`).

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python (%100)
* **Veri İşleme ve Makine Öğrenmesi:** Pandas, NumPy, Scikit-Learn 
* **Model Kaydı:** Pickle

## 💻 Kurulum ve Kullanım

1. Bu depoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/sinann-emree/car-price-prediction-ml.git](https://github.com/sinann-emree/car-price-prediction-ml.git)
