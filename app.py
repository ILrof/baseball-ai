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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"weight_rate": "エラー", "ai_data": "動画がありません"})
        
    video_file = request.files['video']
    temp_path = "temp_video.mp4"
    video_file.save(temp_path)

    rates = []
    cap = cv2.VideoCapture(temp_path)

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
                    rates.append(max(0.0, min(100.0, rate)))

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not rates:
        return jsonify({"weight_rate": "エラー", "ai_data": "骨格検出失敗"})

    # 元の85%固定になる計算ロジック
    impact_rate = rates[int(len(rates) * 0.6)] if len(rates) > 5 else rates[0]

    ai_advice = "AIアドバイス生成失敗"
    if GOOGLE_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(f"野球のバッティングで体重移動率が{impact_rate:.1f}%でした。短くアドバイスをHTML形式のpタグでください。")
            ai_advice = response.text
        except Exception:
            pass

    return jsonify({
        "weight_rate": f"{int(impact_rate)}",
        "ai_data": ai_advice,
        "chart_data": {"labels": [str(i) for i in range(len(rates))], "values": rates}
    })

if __name__ == '__main__':
    app.run(debug=True, port=10000)
