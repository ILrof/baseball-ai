# 1. 安定したPython 3.11環境をベースにする
FROM python:3.11-slim

# 2. 最新のLinuxシステムに適合したOpenCV・MediaPipe用パッケージをインストール
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. フォルダの準備とファイルのコピー
WORKDIR /app
COPY . /app

# 4. パッケージをインストール
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. ポート5001で起動
EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
