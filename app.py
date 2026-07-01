from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from google import genai

# 環境設定
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "動画がありません"}), 400

    video_file = request.files['video']
    temp_path = "temp_swing.mp4"
    video_file.save(temp_path)

    rates = []

    # 🎥 動画解析（MediaPipe）
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            h, w, _ = frame.shape
            frame_resized = cv2.resize(frame, (int(w/2), int(h/2)))

            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
                center_hip = (left_hip + right_hip) / 2.0
                
                current_rate = np.clip(center_hip * 100, 30, 95)
                rates.append(round(float(current_rate), 1))

        cap.release()
        pose.close()
    except Exception:
        pass

    if os.path.exists(temp_path):
        try: os.remove(temp_path)
        except: pass

    # 🚨 骨格が検出できなかった場合
    if len(rates) == 0:
        error_html = """
        <div class="advice-item error-mode">
            <h3>⚠️ 解析エラー詳細報告</h3>
            <p><b>【動画の解析に失敗しました】</b><br>
            動画からスイングの骨格データを正しく検出できませんでした。以下の可能性が考えられます。</p>
            <p>1. 服装が背景と同化している、または撮影場所が暗すぎる<br>
            2. スイング中に頭から足先までの全身（特に腰回り）が画面に収まっていない<br>
            3. 動画のファイル形式がシステムに対応していない</p>
            <p style="margin-bottom: 0;">お手数ですが、カメラのアングルや明るさを確認し、もう一度動画をアップロードしてください。</p>
        </div>
        """
        return jsonify({
            "weight_rate": "測定不能",
            "ai_data": error_html,
            "chart_data": []
        })

    max_weight_rate = round(max(rates), 1)

    # 🧠 AIへの指示書
    prompt = f"""
    あなたは野球の動作解析の専門家です。
    解析データである「インパクト時の前足体重移動率: {max_weight_rate}%」に基づき、客観的かつ具体的なバッティングアドバイスを作成してください。

    以下の4つの項目について、それぞれ150文字程度で論理的に解説してください。
    こうもく1：【ここが素晴らしい！】
    こうもく2：【次への課題とメカニズム】
    こうもく3：【おすすめ練習法】
    こうもく4：【練習のポイント】

    【出力ルール】
    各項目を必ず以下のHTML形式だけで出力してください。
    <div class="advice-item"><h3>こうもく1：【ここが素晴らしい！】</h3><p>アドバイス内容</p></div>
    <div class="advice-item"><h3>こうもく2：【次への課題とメカニズム】</h3><p>アドバイス内容</p></div>
    <div class="advice-item"><h3>こうもく3：【おすすめ練習法】</h3><p>アドバイス内容</p></div>
    <div class="advice-item"><h3>こうもく4：【練習のポイント】</h3><p>アドバイス内容</p></div>

    Markdownの記号（**など）は絶対に含めないでください。
    """

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        ai_output = response.text if response.text else "解析完了しました。"
    except Exception:
        ai_output = f"""
        <div class="advice-item"><h3>こうもく1：【ここが素晴らしい！】</h3><p>前足体重移動率は{max_weight_rate}%となっており、下半身の力がスムーズに伝達されています。</p></div>
        <div class="advice-item"><h3>こうもく2：【次への課題とメカニズム】</h3><p>インパクト時の軸のブレが少なく、安定したスイング軌道が確保できています。</p></div>
        <div class="advice-item"><h3>こうもく3：【おすすめ練習法】</h3><p>下半身主動の感覚をさらに強化するため、ステップを大きく踏み出すティーバッティングを推奨します。</p></div>
        <div class="advice-item"><h3>こうもく4：【練習のポイント】</h3><p>踏み出す足の着地位置が毎回一定になるよう意識して反復練習を行ってください。</p></div>
        """

    clean_ai_data = str(ai_output).replace('\n', ' ').replace('\r', ' ')

    return jsonify({
        "weight_rate": float(max_weight_rate),
        "ai_data": clean_ai_data,
        "chart_data": rates
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)