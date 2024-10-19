from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model yang sudah disimpan menggunakan pickle
with open('knn_model_k3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Tangkap data input dari form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Prediksi menggunakan model yang sudah diload
    prediction = model.predict(final_features)[0]  # Ambil hasil prediksi

    # Konversi hasil prediksi ke string dan berikan keterangan
    if prediction == 0:
        result = 'BAIK'
        keterangan = "Tingkat kualitas udara yang sangat baik, tidak memberikan efek negatif terhadap manusia, hewan, dan tumbuhan. Sangat baik melakukan kegiatan di luar ruangan."
    elif prediction == 1:
        result = 'SANGAT TIDAK SEHAT'
        keterangan = "Tingkat kualitas udara yang dapat meningkatkan risiko kesehatan pada sejumlah segmen populasi yang terpapar. Untuk kelompok sensitif diharapkan dapat menghindari semua aktivitas di luar. Perbanyak aktivitas di dalam ruangan atau lakukan penjadwalan ulang pada waktu dengan kualitas udara yang baik. Dan untuk setiap orang, hindari aktivitas fisik yang terlalu lama di luar ruangan, pertimbangkan untuk melakukan aktivitas di dalam ruangan."
    elif prediction == 2:
        result = 'SEDANG'
        keterangan = "Tingkat kualitas udara masih dapat diterima pada kesehatan manusia, hewan, dan tumbuhan. Pada kondisi ini setiap orang masih dapat beraktivitas di luar. Adapun kelompok sensitif agar mengurangi aktivitas fisik yang terlalu lama atau berat."
    elif prediction == 3:
        result = 'TIDAK SEHAT'
        keterangan = "Tingkat kualitas udara yang bersifat merugikan pada manusia, hewan, dan tumbuhan. Kelompok sensitif masih diperbolehkan untuk melakukan aktivitas di luar, tetapi harus mengambil rehat lebih sering dan melakukan aktivitas ringan. Amati gejala berupa batuk atau nafas sesak."
    else:
        result = "Kualitas udara tidak dapat diidentifikasi. Silakan coba lagi."
        keterangan = ""

    # Arahkan ke halaman hasil dengan keterangan
    return render_template('result.html', prediction_text=f'Hasil prediksi: {result}', keterangan=keterangan)

if __name__ == "__main__":
    app.run(debug=True)
