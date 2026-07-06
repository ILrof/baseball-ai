import os
import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"weight_rate": "測定不能", "ai_data": "<p>動画ファイルが見つかりません。</p>", "chart_data": []})
        
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"weight_rate": "測定不能", "ai_data": "<p>動画が選択されていません。</p>", "chart_data": []})

    temp_path = "temp_video.mp4"
    video_file.save(temp_path)

    rates = []
    cap = cv2.VideoCapture(temp_path)

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * (640 / w))))

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                    center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4.0
                    
                    left_a = left_ankle.x
                    right_a = right_ankle.x
                    
                    if abs(left_a - right_a) > 0.01:
                        if left_a > right_a:
                            rate = ((center_x - right_a) / (left_a - right_a)) * 100
                        else:
                            rate = ((center_x - left_a) / (right_a - left_a)) * 100
                        
                        rate = max(0.0, min(100.0, rate))
                        rates.append(float(rate))

        cap.release()

    except Exception as e:
        if cap.isOpened():
            cap.release()
        error_str = str(e)
        print(f"Error during video processing: {error_str}")
        
        if "timeout" in error_str.lower() or "connection" in error_str.lower():
            error_message = "サーバーとの通信がタイムアウトしました。動画を1〜2秒にするかLINE等で圧縮してください。"
        elif "video" in error_str.lower() or "open" in error_str.lower():
            error_message = "動画ファイルを正しく読み込めませんでした。"
        else:
            error_message = f"解析エラー: {error_str}"

        error_html = f"<div class='advice-item error-mode'><h3>⚠️ 解析エラー</h3><p>{error_message}</p></div>"
        return jsonify({"weight_rate": "測定不能", "ai_data": error_html, "chart_data": []})

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if len(rates) < 3:
        error_html = "<div class='advice-item error-mode'><h3>⚠️ データ不足</h3><p>全身が映るように撮影してください。</p></div>"
        return jsonify({"weight_rate": "測定不能", "ai_data": error_html, "chart_data": []})

    start_pos = rates[0]
    max_pos = max(rates)
    min_pos = min(rates)
    end_pos = rates[-1]
    
    # 💡 85%固定を完全に撃破！スイング中の最大移動率（一番踏み込んだ瞬間）をリアルに表示します
    display_rate = max_pos

    ai_advice = "<p>AIアドバイスの生成に失敗しました。</p>"
    if GOOGLE_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"""
            プロの野球バッティングコーチとして、以下のスイングデータ（体重移動の推移）を分析し、熱血かつ具体的な指導レポートを作成してください。
            【スイングデータ】
            ・始動時: {start_pos:.1f}%
            ・最大移動: {max_pos:.1f}%
            ・最小移動: {min_pos:.1f}%
            ・フォロースルー: {end_pos:.1f}%
            ※0%＝キャッチャー寄り、100%＝ピッチャー寄り
            HTMLタグ（<h3>や<p>など）を使って出力してください。
            """
            response = model.generate_content(prompt)
            if response.text:
                ai_advice = response.text
        except Exception as ai_err:
            ai_advice = f"<p>AI解析エラー: {str(ai_err)}</p>"

    chart_data = {
        "labels": [f"{i+1}" for i in range(len(rates))],
        "values": rates
    }

    return jsonify({
        "weight_rate": f"{int(display_rate)}",
        "ai_data": ai_advice,
        "chart_data": chart_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=10000)
