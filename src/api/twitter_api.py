import os
import requests
from dotenv import load_dotenv
import csv

# load token dari .env
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def get_tweets(query, max_results=20):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,text"
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data

def save_to_csv(data, filename="tweets_gempa.csv"):
    tweets = data.get("data", [])
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["created_at", "text"])
        for t in tweets:
            writer.writerow([t["created_at"], t["text"]])
    print("DONE → CSV berhasil dibuat:", filename)

if __name__ == "__main__":
    print("Mengambil tweet...")
    data = get_tweets("gempa")
    save_to_csv(data)

