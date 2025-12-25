import argparse
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Prediksi Status Stunting Menggunakan Model KNN")
    parser.add_argument("--data", type=str, required=True,
                        help="Masukkan 3 fitur: Umur (bulan), Tinggi Badan (cm), Berat Badan (kg) — dipisahkan dengan spasi")
    args = parser.parse_args()

    # Parse input
    try:
        values = list(map(float, args.data.strip().split()))
        if len(values) != 3:
            print("Input harus terdiri dari tepat 3 nilai: Umur TinggiBadan BeratBadan")
            return

        # Buat DataFrame dengan nama kolom sesuai training
        input_df = pd.DataFrame(
            [values],
            columns=["Umur (bulan)", "Tinggi Badan (cm)", "Berat Badan (kg)"]
        )
    except ValueError:
        print("Input tidak valid: Pastikan semua fitur berupa angka dan dipisahkan dengan spasi.")
        return

    # Load model dan preprocessor
    try:
        model = joblib.load("D:\\deploy\\KNeighborsClassifierModel.pkl")
        scaler = joblib.load("D:\\deploy\\Preprocessor.pkl")
    except Exception as e:
        print(f"Error saat memuat model atau scaler: {e}")
        return

    # Transformasi input
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        print(f"Error saat menormalkan data input: {e}")
        return

    # Prediksi
    try:
        hasil = model.predict(input_scaled)
        status = "Stunted" if hasil[0] == 1 else "Normal"
        print(f"Status Gizi (Stunting): {status}")
    except Exception as e:
        print(f"Error saat prediksi: {e}")

if __name__ == "__main__":
    main()

# data
# python predict_stunting.py --data "24 80 10.5"
