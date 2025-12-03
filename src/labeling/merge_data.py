import pandas as pd

# Baca dataset bersih dari Kaggle
df_kaggle = pd.read_csv("kaggle_clean.csv")

# Baca dataset bersih dari Twitter API
df_twitter = pd.read_csv("twitter_clean.csv")

# Pastikan kolom 'text_clean' ada
if "text_clean" not in df_kaggle.columns:
    raise Exception("Kolom 'text_clean' tidak ada di kaggle_clean.csv")

if "text_clean" not in df_twitter.columns:
    raise Exception("Kolom 'text_clean' tidak ada di twitter_clean.csv")

# Dataset Kaggle sudah punya label 'target'
# Dataset Twitter API belum punya label -> asumsikan target = 1 (tweet bencana)
df_twitter["target"] = 1

# Ambil kolom penting saja
df_kaggle_final = df_kaggle[["text_clean", "target"]]
df_twitter_final = df_twitter[["text_clean", "target"]]

# Gabungkan dua dataset
df_final = pd.concat([df_kaggle_final, df_twitter_final], ignore_index=True)

# Simpan hasil sebagai dataset_final.csv
df_final.to_csv("dataset_final.csv", index=False)

print("[DONE] dataset_final.csv berhasil dibuat!")
print("Total data:", len(df_final))
