# 1. パソコン環境に一番近い、安定したPython 3.11のLinux環境をベースにする
FROM python:3.11-slim

# 2. MediaPipeとOpenCVが動くために絶対に不可欠なLinuxの部品を強制インストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. サーバー内の作業フォルダを決定
WORKDIR /app

# 4. あなたのプログラム一式をサーバーにコピー
COPY . /app

# 5. pipを最新にしてから、必要な部品をインストール
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 6. ポート5001でWebアプリ（Gunicorn）を起動
EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
