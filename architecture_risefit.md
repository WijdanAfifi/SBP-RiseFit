# RiseFit — System Architecture

## 1. Overview

**RiseFit** adalah aplikasi kebugaran yang menggabungkan analisis kondisi tubuh, rekomendasi olahraga berbasis aturan, dan deteksi gerakan untuk menghitung repetisi olahraga secara otomatis.

RiseFit menggunakan dua pendekatan utama:

- **Knowledge-Based System (KBS):** menghasilkan rekomendasi olahraga melalui Knowledge Base, aturan IF–THEN, dan metode inferensi Forward Chaining.
- **Computer Vision:** menggunakan model YOLO yang mendukung pose estimation untuk memperoleh keypoints tubuh saat pengguna melakukan latihan. Keypoints diproses oleh logika penghitung repetisi.

Rekomendasi makanan dan grafik progress direncanakan sebagai fitur opsional, bergantung pada waktu dan kapasitas pengembangan.

## 2. Project Scope

### 2.1 Fitur utama

1. Klasifikasi kondisi tubuh berdasarkan BMI.
2. Rekomendasi olahraga berdasarkan data dan tujuan pengguna.
3. Deteksi pose saat melakukan olahraga.
4. Penghitungan repetisi untuk gerakan yang didukung.

### 2.2 Fitur opsional

1. Rekomendasi makanan berdasarkan tujuan dan preferensi pengguna.
2. Grafik progress berat badan dan aktivitas olahraga.

### 2.3 Di luar cakupan MVP

- Diagnosis medis atau pengukuran langsung persentase lemak tubuh.
- Pemantauan kesehatan real-time untuk keperluan medis.
- Integrasi smartwatch atau platform kesehatan eksternal.
- Social/community, pembayaran, dan layanan personal trainer.
- Machine learning khusus yang dilatih dari nol, kecuali diperlukan dan disetujui dalam pengembangan lanjutan.

## 3. Architectural Style

RiseFit menggunakan arsitektur berlapis (*layered architecture*) agar antarmuka pengguna, logika aplikasi, sistem rekomendasi, computer vision, dan penyimpanan data dapat dikembangkan secara terpisah.

```text
┌─────────────────────────────────────────────────┐
│                 PRESENTATION LAYER              │
│  Login/Register • Dashboard • Profile • Program │
│  Exercise Camera/Counter • Progress (optional)  │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│                 APPLICATION LAYER               │
│  Authentication • User Profile • BMI Analysis   │
│  Recommendation Service • Exercise Session      │
│  Progress Service (optional)                    │
└───────────────┬─────────────────────┬───────────┘
                │                     │
                ▼                     ▼
┌──────────────────────────┐  ┌─────────────────────────┐
│ KNOWLEDGE / INFERENCE     │  │ COMPUTER VISION         │
│ Knowledge Base            │  │ Camera Frame Processing  │
│ IF–THEN Rules             │  │ YOLO Pose Estimation     │
│ Forward Chaining          │  │ Body Keypoints           │
│ Exercise Recommendations  │  │ Rep Counter              │
└───────────────┬──────────┘  └────────────┬────────────┘
                │                          │
                └────────────┬─────────────┘
                             ▼
┌─────────────────────────────────────────────────┐
│                    DATA LAYER                   │
│ Users • Body Profiles • Goals • Rules           │
│ Exercises • Exercise Sessions • Progress        │
│ Foods (optional) • Food Recommendations (opt.)  │
└─────────────────────────────────────────────────┘
```

## 4. High-Level User Flow

```text
START
  |
  v
Login / Register
  |
  v
Input or Update User Profile
(age, gender, height, weight, goal)
  |
  v
Dashboard
  |
  +----------------------+-----------------------+
  |                      |                       |
  v                      v                       v
Body Status          Program                 Exercise
(BMI)                Recommendation           Monitoring
  |                      |                       |
  |                      v                       v
  |                Knowledge Base            Camera Input
  |                      |                       |
  |                      v                       v
  |                Forward Chaining          YOLO Pose
  |                      |                       |
  |                      v                       v
  |                Exercise Plan             Body Keypoints
  |                                              |
  |                                              v
  |                                         Rep Counter
  |                                              |
  +----------------------+-----------------------+
                         |
                         v
                  Results / History
                         |
                         v
              Progress Graph (optional)
```

## 5. Main Components

### 5.1 Authentication

Mengelola pembuatan akun, login, logout, dan sesi pengguna. Halaman yang menampilkan atau mengubah data pribadi hanya dapat diakses setelah autentikasi berhasil.

### 5.2 User Profile

Menyimpan data dasar yang diperlukan oleh fitur RiseFit:

- Umur
- Gender
- Tinggi badan
- Berat badan
- Tujuan kebugaran
- Pengalaman olahraga (digunakan untuk rekomendasi jika diimplementasikan)

Data profil dapat diperbarui oleh pengguna. Perubahan data dapat memicu perhitungan ulang BMI dan digunakan dalam rekomendasi berikutnya.

### 5.3 BMI and Body-Status Analysis

BMI dihitung menggunakan rumus:

```text
BMI = weight_kg / (height_m * height_m)
```

Sistem mengelompokkan nilai BMI menggunakan batas kategori dewasa yang ditetapkan pada konfigurasi aplikasi. BMI adalah indikator skrining awal; BMI tidak secara langsung mendeteksi bentuk tubuh, massa otot, atau persentase lemak dan bukan diagnosis medis.

### 5.4 Recommendation Service

Menghasilkan rekomendasi olahraga berdasarkan data profil, tujuan pengguna, dan aturan dalam Knowledge Base.

Alur proses:

```text
User Facts
    |
    v
BMI / Profile Analysis
    |
    v
Knowledge Base (IF–THEN Rules)
    |
    v
Forward Chaining
    |
    v
Selected Exercise Recommendations
    |
    v
Program Page
```

### 5.5 Knowledge Base

Knowledge Base menyimpan fakta dan aturan yang dipakai sistem untuk memilih rekomendasi. Kelompok pengetahuan yang direncanakan:

- **Body Knowledge:** aturan klasifikasi BMI.
- **Goal Knowledge:** aturan berdasarkan tujuan pengguna.
- **Exercise Knowledge:** jenis, kategori, dan tingkat kesulitan olahraga.
- **Experience Knowledge:** aturan penyesuaian berdasarkan pengalaman olahraga.
- **Nutrition Knowledge (optional):** strategi makanan dan pilihan makanan.

Contoh aturan:

```text
IF goal = "Muscle Gain"
AND experience = "Beginner"
THEN exercise_recommendation = "Beginner Strength Training"
```

Aturan dan rekomendasi harus ditinjau sebelum digunakan dan tidak boleh dipresentasikan sebagai saran medis.

### 5.6 Inference Engine — Forward Chaining

Inference Engine memulai proses dari fakta yang tersedia tentang pengguna, kemudian mengecek aturan IF–THEN yang kondisinya terpenuhi. Kesimpulan yang dihasilkan digunakan untuk memilih rekomendasi.

```text
Input Facts
    |
    v
Find Applicable Rules
    |
    v
Evaluate Conditions
    |
    v
Produce Conclusions
    |
    v
Return Recommendations
```

### 5.7 Exercise Monitoring

Modul ini memproses kamera atau video ketika pengguna melakukan gerakan yang didukung.

Tahapan:

1. Mengambil frame dari kamera atau video.
2. Mengirim frame ke model YOLO yang mendukung pose estimation.
3. Mendapatkan keypoints tubuh dan nilai confidence.
4. Memeriksa apakah keypoints cukup jelas untuk diproses.
5. Menghitung sudut atau posisi yang diperlukan untuk gerakan tersebut.
6. Menggunakan state machine atau aturan transisi untuk menghitung repetisi.
7. Menampilkan hitungan repetisi dan status sesi.

```text
Camera / Video
      |
      v
Frame Processing
      |
      v
YOLO Pose Estimation
      |
      v
Body Keypoints + Confidence
      |
      v
Movement / Angle Calculation
      |
      v
Rep-Counting State Machine
      |
      v
Repetition Count
```

YOLO Pose menyediakan estimasi posisi tubuh; penghitungan repetisi dilakukan oleh logika tambahan yang menganalisis perubahan pose antar-frame. Model deteksi objek YOLO biasa tidak otomatis menyediakan keypoints, sehingga varian model yang digunakan harus mendukung pose estimation.

### 5.8 Rep Counter

Rep Counter memelihara keadaan gerakan, misalnya `UP`, `DOWN`, atau `UNKNOWN`, sesuai latihan yang sedang dianalisis. Satu repetisi hanya dihitung ketika urutan transisi yang ditentukan untuk latihan tersebut selesai.

Contoh konseptual untuk squat:

```text
Standing / UP
      |
      v
Squat Down / DOWN
      |
      v
Return to Standing / UP
      |
      v
Increment Repetition Count
```

Ambang sudut dan aturan transisi harus dikalibrasi serta diuji untuk setiap latihan. Jika pose tidak terlihat jelas atau confidence rendah, sistem sebaiknya tidak menambah hitungan.

### 5.9 Food Recommendation (Optional)

Jika fitur ini dikerjakan, sistem menampilkan rekomendasi makanan umum berdasarkan tujuan dan preferensi pengguna melalui Knowledge Base dan aturan IF–THEN. Fitur ini tidak memerlukan YOLO. Hindari klaim diagnosis atau rencana diet medis.

### 5.10 Progress Tracking (Optional)

Jika fitur ini dikerjakan, sistem menyimpan riwayat berat badan dan/atau hasil sesi olahraga, lalu menampilkannya sebagai grafik. Data dapat mencakup tanggal, berat badan, jenis latihan, dan jumlah repetisi. Analisis tren harus didasarkan pada data historis yang tersedia dan tidak menjanjikan hasil tubuh di masa depan.

## 6. Data Architecture

Entitas awal yang disarankan:

| Entity | Purpose | Core fields (illustrative) |
|---|---|---|
| `users` | Data akun | `user_id`, `username`, `password_hash`, timestamps |
| `body_profiles` | Data kondisi dan tujuan | `profile_id`, `user_id`, `age`, `gender`, `height`, `weight`, `goal` |
| `exercises` | Katalog latihan | `exercise_id`, `name`, `category`, `difficulty`, `pose_config` |
| `rules` | Aturan Knowledge Base | `rule_id`, `rule_code`, `rule_type`, `condition`, `conclusion`, `priority`, `is_active` |
| `recommendations` | Rekomendasi yang dihasilkan | `recommendation_id`, `user_id`, `rule_id`, `created_at` |
| `exercise_sessions` | Sesi deteksi olahraga | `session_id`, `user_id`, `exercise_id`, `started_at`, `ended_at`, `repetition_count` |
| `progress_records` | Catatan progress (optional) | `progress_id`, `user_id`, `recorded_at`, `weight`, `notes` |
| `foods` | Katalog makanan (optional) | `food_id`, `name`, `category`, nutrition fields if verified |

Struktur final dan kardinalitas tabel harus mengikuti ERD yang disepakati. Jika satu rekomendasi berisi banyak olahraga, gunakan tabel penghubung atau tabel item rekomendasi agar relasi tersimpan dengan benar. Kata sandi wajib disimpan sebagai hash, bukan teks biasa.

## 7. Main Data Flow

### 7.1 Recommendation Flow

```text
Profile Data
    -> BMI Analysis
    -> User Facts
    -> Knowledge Base
    -> Forward Chaining
    -> Exercise Recommendation
    -> Program UI
```

### 7.2 Exercise Detection Flow

```text
Exercise Selection
    -> Camera Permission
    -> Frame Capture
    -> YOLO Pose
    -> Keypoint Validation
    -> Movement State Detection
    -> Rep Counter
    -> Session Result
```

### 7.3 Optional Progress Flow

```text
User / Session Data
    -> Save Progress Record
    -> Retrieve History
    -> Calculate Changes
    -> Render Progress Graph
```

## 8. Interface Modules

Antarmuka utama yang direncanakan:

- **Login:** masuk ke akun.
- **Register:** membuat akun baru.
- **Dashboard:** ringkasan BMI/kategori kondisi tubuh dan navigasi.
- **Profile:** melihat dan mengubah data pengguna.
- **Program:** melihat rekomendasi olahraga.
- **Exercise Monitoring:** memilih latihan, mengaktifkan kamera, dan melihat hitungan repetisi.
- **Progress (optional):** melihat riwayat dan grafik perkembangan.
- **Food Recommendation (optional):** melihat rekomendasi makanan.

## 9. Non-Functional Requirements

### 9.1 Performance

- Antarmuka harus tetap responsif saat data profil dan rekomendasi diproses.
- Frame kamera diproses dengan mempertimbangkan kemampuan perangkat.
- Jika proses pose estimation lambat, UI harus tetap menampilkan status pemrosesan dan tidak membekukan seluruh aplikasi.

### 9.2 Reliability

- Hitungan repetisi hanya bertambah jika urutan gerakan memenuhi aturan yang ditetapkan.
- Pose yang tidak terdeteksi dengan cukup baik tidak boleh otomatis dianggap sebagai repetisi valid.
- Kesalahan kamera atau model harus ditampilkan secara jelas.

### 9.3 Security and Privacy

- Password disimpan menggunakan hashing yang aman.
- Data profil dan riwayat hanya dapat diakses oleh pemilik akun yang terautentikasi.
- Akses kamera meminta izin pengguna.
- Frame atau video tidak disimpan secara permanen kecuali penyimpanan tersebut memang diperlukan, dijelaskan kepada pengguna, dan disetujui.

### 9.4 Maintainability

- Aturan rekomendasi disimpan terpisah dari kode tampilan.
- Konfigurasi pose dan ambang gerakan dibuat agar dapat diubah serta diuji per jenis latihan.
- Katalog latihan dapat diperluas tanpa merombak seluruh aplikasi.

### 9.5 Usability

- Pengguna memperoleh instruksi posisi kamera dan gerakan yang didukung.
- Hitungan repetisi terlihat jelas.
- Pesan kesalahan menjelaskan tindakan yang dapat dilakukan pengguna.

## 10. Validation and Error Handling

Validasi minimum:

- Umur harus berupa nilai yang masuk akal.
- Tinggi dan berat badan harus bernilai positif.
- Tujuan harus berasal dari pilihan yang didukung.
- Data login dan registrasi wajib divalidasi.
- Kamera harus mendapat izin sebelum digunakan.
- Keypoints harus memenuhi pemeriksaan confidence dan kelengkapan sebelum digunakan untuk menghitung repetisi.
- Data progress, jika tersedia, harus memiliki tanggal dan nilai yang valid.

Kondisi yang perlu ditangani:

- Login gagal atau data tidak valid.
- Kamera tidak tersedia atau izin ditolak.
- Tidak ada orang yang terdeteksi.
- Pose terhalang atau keypoints tidak cukup jelas.
- Latihan yang dilakukan tidak termasuk latihan yang didukung.

## 11. Suggested Technology Boundaries

Pilihan teknologi final ditentukan oleh tim. Batas tanggung jawab yang disarankan:

| Area | Responsibility |
|---|---|
| Frontend | UI, form, dashboard, visualisasi, akses kamera |
| Backend / application logic | Authentication, profil, penyimpanan data, API |
| Knowledge-Based System | Knowledge Base, rule matching, Forward Chaining |
| Computer Vision module | YOLO Pose inference dan ekstraksi keypoints |
| Rep Counter | State machine, perhitungan transisi, hitungan repetisi |
| Database | Data akun, profil, aturan, katalog latihan, sesi, dan progress |

Modul computer vision dapat berjalan di backend atau pada perangkat pengguna, tergantung performa, framework, dan kebutuhan deployment. Keputusan ini harus dibuat sebelum integrasi.

## 12. MVP Acceptance Criteria

MVP dianggap memenuhi fungsi utamanya apabila:

1. Pengguna dapat mendaftar, login, logout, dan mengelola profil.
2. Sistem dapat menghitung BMI dan menampilkan kategorinya dengan penjelasan bahwa BMI bukan diagnosis medis.
3. Sistem menghasilkan rekomendasi olahraga dari aturan yang terdokumentasi.
4. Pengguna dapat memilih latihan yang didukung dan memulai pemrosesan kamera.
5. Model pose menghasilkan keypoints yang dapat dipakai oleh penghitung repetisi.
6. Repetisi dihitung berdasarkan urutan transisi gerakan yang telah ditentukan dan diuji.
7. Sistem menangani kondisi kamera atau pose yang tidak memadai tanpa menambah hitungan secara sembarangan.
8. Rekomendasi makanan dan grafik progress dapat ditambahkan sebagai fitur opsional tanpa menghambat fitur inti.

## 13. Design Principles

- **Separation of concerns:** UI, aturan rekomendasi, computer vision, penghitung repetisi, dan database memiliki tanggung jawab terpisah.
- **Explainable recommendations:** rekomendasi dapat ditelusuri ke aturan yang memicunya.
- **Modular computer vision:** konfigurasi gerakan dan penghitung repetisi dapat diuji per latihan.
- **Privacy by design:** kamera dan data pengguna digunakan sesuai izin serta kebutuhan fitur.
- **Incremental development:** mulai dari satu gerakan yang dapat dihitung dengan baik sebelum menambah gerakan lain.

## 14. Conclusion

RiseFit menggabungkan analisis BMI, rekomendasi olahraga berbasis Knowledge-Based System, serta pemantauan gerakan menggunakan YOLO Pose dan algoritma penghitung repetisi.

Komponen utama sistem adalah:

```text
User Profile
    -> BMI Analysis
    -> Knowledge Base + Forward Chaining
    -> Exercise Recommendation

Camera
    -> YOLO Pose
    -> Body Keypoints
    -> Rep-Counting State Machine
    -> Exercise Session Result
```

Rekomendasi makanan dan grafik progress menjadi fitur opsional. Arsitektur ini merupakan blueprint awal dan perlu diselaraskan dengan ERD, keputusan teknologi, serta pengujian model pose yang dipilih.
