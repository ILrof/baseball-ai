# 1. 💡スリム版をやめ、すべての部品が最初から全部入りになっているフル版Python3.11を使う
FROM python:3.11

# 2. 最初から全部入っているため、apt-getでのややこしいインストール作業は一切不要！
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 3. フォルダ準備とファイルコピー
WORKDIR /app
COPY . /app

# 4. パッケージのインストール（確実に動く組み合わせ）
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. ポート5001で起動
EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
