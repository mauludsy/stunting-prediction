# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import warnings

# warnings.filterwarnings("ignore")

# app = Flask(__name__)

# # Load model dan scaler
# model_path = "D:/deploy/KNeighborsClassifierModel.pkl"
# scaler_path = "D:/deploy/Preprocessor.pkl"

# try:
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
# except Exception as e:
#     print(f"Gagal memuat model atau preprocessor: {e}")
#     raise

# @app.route("/", methods=["GET"])
# def home():
#     return "API Prediksi Status Gizi Anak - KNN Model", 200

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         # Validasi input
#         required_fields = ["umur", "tinggi_badan", "berat_badan"]
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({"error": f"Field '{field}' wajib disertakan."}), 400

#         # Ambil nilai input
#         umur = float(data["umur"])
#         tinggi = float(data["tinggi_badan"])
#         berat = float(data["berat_badan"])

#         # Buat DataFrame sesuai dengan kolom saat training
#         input_df = pd.DataFrame([{
#             "Umur (bulan)": umur,
#             "Tinggi Badan (cm)": tinggi,
#             "Berat Badan (kg)": berat
#         }])

#         # Scaling
#         input_scaled = scaler.transform(input_df)

#         # Prediksi
#         prediction = model.predict(input_scaled)[0]

#         # Mapping hasil prediksi ke label
#         label_map = {0: "Normal", 1: "Stunted"}
#         hasil_prediksi = label_map.get(prediction, str(prediction))

#         return jsonify({
#             "umur": umur,
#             "tinggi_badan": tinggi,
#             "berat_badan": berat,
#             "prediksi_status_gizi": hasil_prediksi
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Pastikan Anda sudah 'pip install Flask-Cors'
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk mengizinkan request dari frontend

# Load model dan scaler
# PASTIKAN JALUR INI BENAR PADA SISTEM ANDA
model_path = "D:/deploy/KNeighborsClassifierModel.pkl"
scaler_path = "D:/deploy/Preprocessor.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model dan preprocessor berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model atau preprocessor: {e}")
    # Penting: Jika model/scaler tidak bisa dimuat, aplikasi tidak bisa berfungsi.
    # Sebaiknya hentikan aplikasi atau log error secara lebih serius.
    raise

@app.route("/", methods=["GET"])
def home():
    return "API Prediksi Status Gizi Anak - KNN Model", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"Menerima data dari frontend: {data}") # Untuk debugging di konsol backend

        # Ambil nilai 'nama' (opsional untuk prediksi, tapi akan dikembalikan)
        nama = data.get("nama", "Tidak Diketahui") # Jika 'nama' tidak ada, default ke "Tidak Diketahui"

        # Validasi field yang WAJIB untuk model prediksi
        required_fields_for_model = ["umur", "tinggi_badan", "berat_badan"]
        for field in required_fields_for_model:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Field '{field}' wajib disertakan dan tidak boleh kosong."}), 400
            
            # Coba konversi ke float dan tangani ValueError
            try:
                data[field] = float(data[field])
            except ValueError:
                return jsonify({"error": f"Nilai '{field}' harus berupa angka yang valid."}), 400

        # Ambil nilai input setelah validasi dan konversi
        umur = data["umur"]
        tinggi = data["tinggi_badan"]
        berat = data["berat_badan"]

        # Buat DataFrame sesuai dengan kolom saat training model
        # Perhatikan: 'nama' TIDAK disertakan di sini karena model Anda tidak dilatih dengan fitur nama.
        input_df = pd.DataFrame([{
            "Umur (bulan)": umur,
            "Tinggi Badan (cm)": tinggi,
            "Berat Badan (kg)": berat
        }])
        print(f"DataFrame input untuk scaling: {input_df}")

        # Scaling data input
        input_scaled = scaler.transform(input_df)
        print(f"Input scaled: {input_scaled}")

        # Prediksi menggunakan model
        prediction = model.predict(input_scaled)[0]
        print(f"Raw prediction from model: {prediction}")

        # Mapping hasil prediksi ke label yang mudah dibaca
        label_map = {0: "Normal", 1: "Stunted"} # Sesuaikan dengan mapping label model Anda
        hasil_prediksi = label_map.get(prediction, "Status Tidak Diketahui") # Default jika hasil prediksi di luar 0/1

        print(f"Hasil Prediksi untuk {nama}: {hasil_prediksi}") # Logging hasil prediksi dengan nama
        return jsonify({
            "nama": nama, # Mengembalikan nama ke frontend
            "umur": umur,
            "tinggi_badan": tinggi,
            "berat_badan": berat,
            "prediksi_status_gizi": hasil_prediksi
        })

    except Exception as e:
        print(f"Terjadi kesalahan di endpoint /predict: {e}") # Log error di server
        return jsonify({"error": str(e), "message": "Terjadi kesalahan internal server. Mohon coba lagi."}), 500

if __name__ == "__main__":
    # app.run akan secara default berjalan di localhost (127.0.0.1)
    # Pastikan port 5000 tidak digunakan oleh aplikasi lain
    app.run(debug=True, port=5000)