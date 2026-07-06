# 1. 安定したPython 3.11のフルパッケージ環境
FROM python:3.11

# 2. OpenCVとMediaPipeの起動に必要なグラフィックシステム部品を確実にインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. フォルダ準備とファイルコピー
WORKDIR /app
COPY . /app

# 4. パッケージのインストール
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. ポート5001で起動
EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
