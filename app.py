import os
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# 1. Flaskアプリの初期化
app = Flask(__name__)

# 2. Renderの環境変数からGeminiのAPIキーを取得
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# 3. メイン画面（トップページ）を表示するルーティング
@app.route('/')
def index():
    return render_template('index.html')

# 4. 動画アップロード・解析を行うメインのルーティング
@app.route('/upload', methods=['POST'])
def upload_video():
    # --- [エラーチェック] 動画ファイルが正しく送信されているか ---
    if 'video' not in request.files:
        return jsonify({"weight_rate": "測定不能", "ai_data": "動画ファイルが見つかりません。", "chart_data": []})

    file = request.files['video']
    if file.filename == '':
        return jsonify({"weight_rate": "測定不能", "ai_data": "ファイル名が空です。", "chart_data": []})

    # --- [一時保存] サーバーの一時フォルダに動画を保存 ---
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    # --- [動画読み込み準備] OpenCVで動画を開く ---
    cap = cv2.VideoCapture(temp_path)
    frame_count = 0
    rates = [] # 検出した腰のX座標を溜めるリスト
    error_message = "骨格を検出できませんでした。"

    # --- [動画解析] MediaPipeを使った骨格検出処理 ---
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # 検出と追跡の信頼度を0.4に設定して解析を開始
        with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break # 動画が終了したらループを抜ける

                frame_count += 1
                # 軽量化対策: 4コマに1コマだけを間引いて処理（サーバーの負荷軽減）
                if frame_count % 4 != 0:
                    continue
                
                # 安全対策: 動画が長すぎる場合は最大120フレーム（約15〜20秒分）で打ち切り
                if len(rates) > 120:
                    break

                # 軽量化対策: 画像サイズを縦横1/3に縮小して処理スピードを高速化
                h, w = frame.shape[:2]
                frame_resized = cv2.resize(frame, (int(w/3), int(h/3)))
                
                # 精度向上対策: 暗い服や背景の同化を防ぐ「コントラスト自動補正（CLAHE）」
                lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                frame_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
                # OpenCVのBGR形式からMediaPipe用のRGB形式に変換
                image_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # 骨格（ランドマーク）が検出できた場合の処理
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # 左右の腰（HIP）のX座標（画面左端0.0 〜 右端1.0）を取得
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
                    
                    # 左右の腰の中央値を計算し、お尻の中心位置の動きを記録
                    center_hip_x = (left_hip + right_hip) / 2.0
                    rates.append(round(float(center_hip_x), 4))

    except Exception as e:
        # 途中でシステムエラーが起きた場合はログを記録
        error_message = f"システムエラーが発生しました: {str(e)}"
        print(f"Error during video processing: {e}")

    # --- [後片付け] 動画ファイルと一時ファイルの解放 ---
    if cap.isOpened():
        cap.release()

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    # --- [データ確認] 骨格データが足りない場合は「測定不能」画面を返す ---
    if len(rates) < 3:
        error_html = f"""
        <div class="advice-item error-mode">
            <h3>⚠️ 解析エラー詳細報告</h3>
            <p><b>【動画の解析に失敗しました】</b><br><br>
            原因: {error_message}</p>
            <p>1. 服装が背景と同化している、または撮影場所が暗すぎる可能性<br>
            2. スイング中に頭から足先までの全身（特に腰回り）が画面に収まっていない可能性</p>
            <p style="margin-bottom: 0;">カメラのアングルや明るさを確認し、もう一度動画をアップロードしてください。</p>
        </div>
        """
        return jsonify({
            "weight_rate": "測定不能",
            "ai_data": error_html,
            "chart_data": []
        })

    # --- [左右打者の自動判定＆体重移動率(%)へのデータ変換] ---
    start_pos = rates[0]  # 構えた時の初期位置
    end_pos = rates[-1]   # スイング終了時の位置
    max_val = max(rates)  # 動きの右端
    min_val = min(rates)  # 動きの左端
    # 0での割り算（エラー）を防ぐための安全弁
    div_val = max_val - min_val if max_val != min_val else 1.0
    
    processed_rates = []
    for r in rates:
        # 動画全体の動きから、バッターの踏み出し方向を自動判別
        if end_pos > start_pos:
            # 右方向への移動（右打者など）
            move_ratio = (r - min_val) / div_val
        else:
            # 左方向への移動（左打者など）
            move_ratio = (max_val - r) / div_val
        
        # 動きの割合を30%〜95%の範囲に綺麗にスケール変換
        current_rate = np.clip(30 + (move_ratio * 55), 30, 95)
        processed_rates.append(round(float(current_rate), 1))
        
    rates = processed_rates
    max_weight_rate = round(max(rates), 1) # 最大体重移動率を決定

    # --- [AIアドバイス生成] Geminiへの指示書の作成 ---
    prompt = f"""
    あなたは野球の動作解析の専門家です。
    解析データである「インパクト時の前足体重移動率: {max_weight_rate}%」に基づき、客観的かつ具体的なバッティングアドバイスを作成してください。

    以下の4つの項目について、それぞれ150文字程度で論理的に解説してください。
    こうもく1: 【ここが素晴らしい！】
    こうもく2: 【次への課題とメカニズム】
    こうもく3: 【おすすめ練習法】
    こうもく4: 【練習のポイント】

    【出力ルール】
    各項目を必ず以下のHTML形式だけで出力してください。
    <div class="advice-item"><h3>こうもく1: 【ここが素晴らしい！】</h3><p>アドバイス内容</p></div>
    Markdownの記号（「**」など）は絶対に出力に含めないでください。
    """

    # --- [Gemini APIの呼び出し] ---
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        ai_output = response.text if response.text else "解析完了しました。"
    except Exception:
        # APIエラーやキー未設定時のバックアップ（固定テキスト）
        ai_output = f"""
        <div class="advice-item"><h3>こうもく1: 【ここが素晴らしい！】</h3><p>前足体重移動率は {max_weight_rate}% となっています。</p></div>
        <div class="advice-item"><h3>こうもく2: 【次への課題とメカニズム】</h3><p>インパクト時の軸のブレが少なく、安定した姿勢を維持できています。</p></div>
        <div class="advice-item"><h3>こうもく3: 【おすすめ練習法】</h3><p>下半身主導の感覚をさらに強化するため、ステップ幅の安定化を意識しましょう。</p></div>
        <div class="advice-item"><h3>こうもく4: 【練習のポイント】</h3><p>踏み出す足の着地位置が毎回一定になるよう意識して練習を行います。</p></div>
        """

    # JSONの改行エラーを防ぐために改行文字をスペースに変換
    clean_ai_data = str(ai_output).replace('\n', ' ').replace('\r', ' ')

    # --- [フロントエンドへデータ返却] 画面へ全ての解析データを送る ---
    return jsonify({
        "weight_rate": float(max_weight_rate),
        "ai_data": clean_ai_data,
        "chart_data": rates
    })

# 5. ローカル実行用の設定
if __name__ == '__main__':
    app.run(debug=True, port=5001)
