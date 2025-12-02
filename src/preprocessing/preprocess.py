import pandas as pd
import re

# ---------- FUNGSI BERSIHIN TEKS ----------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()                          # jadi huruf kecil semua
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # hapus link/url
    text = re.sub(r"@\w+", " ", text)              # hapus @username
    text = re.sub(r"#", " ", text)                 # hapus tanda #
    text = re.sub(r"\d+", " ", text)               # hapus angka
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # buang simbol/tanda baca
    text = re.sub(r"\s+", " ", text).strip()       # rapikan spasi
    return text

# ---------- PROSES DATASET KAGGLE ----------

def preprocess_kaggle():
    print("[INFO] Baca tweets.csv (Kaggle)...")
    df = pd.read_csv("tweets.csv")   # file Kaggle harus bernama tweets.csv

    if "text" not in df.columns:
        raise Exception("Kolom 'text' tidak ada di tweets.csv Kaggle")

    df["text_clean"] = df["text"].apply(clean_text)

    # simpan hasil
    df.to_csv("kaggle_clean.csv", index=False)
    print("[DONE] kaggle_clean.csv berhasil dibuat.")

# ---------- PROSES DATASET TWITTER API ----------

def preprocess_twitter():
    print("[INFO] Baca tweets_gempa.csv (Twitter API)...")
    df = pd.read_csv("tweets_gempa.csv")  # file dari API

    # coba cari kolom teks
    text_col = None
    for c in ["text", "tweet", "full_text"]:
        if c in df.columns:
            text_col = c
            break
    
    if text_col is None:
        raise Exception("Tidak menemukan kolom teks di tweets_gempa.csv")

    df["text_clean"] = df[text_col].apply(clean_text)

    df.to_csv("twitter_clean.csv", index=False)
    print("[DONE] twitter_clean.csv berhasil dibuat.")

# ---------- MAIN ----------

if __name__ == "__main__":
    preprocess_kaggle()
    preprocess_twitter()
    print("[SELESAI] Semua preprocessing selesai.")
