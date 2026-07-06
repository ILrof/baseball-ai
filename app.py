import os
import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Gemini APIの初期設定
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# MediaPipeの初期化
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

    # 一時ファイルとして保存
    temp_path = "temp_video.mp4"
    video_file.save(temp_path)

    rates = []
    cap = cv2.VideoCapture(temp_path)

    try:
        # MediaPipe Poseの読み込み
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 処理高速化のために画像サイズを縮小
                h, w, _ = frame.shape
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * (640 / w))))

                # RGBに変換して骨格検出
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 左右の肩、腰、足首の座標を取得
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                    # 重心（腰と肩の中央）の計算
                    center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4.0
                    
                    # スタンス幅に対する相対的な位置から体重移動率を計算
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
        print(f"Error during video processing: {e}")
        
        # エラー原因ごとに、画面に出す分かりやすい日本語メッセージを切り分ける
        if "timeout" in error_str.lower() or "connection" in error_str.lower():
            error_message = "サーバーとの通信がタイムアウトしました。動画の長さを1〜2秒程度にするか、LINE等で圧縮して容量を軽くしてから再度お試しください。"
        elif "video" in error_str.lower() or "open" in error_str.lower() or "codec" in error_str.lower():
            error_message = "動画ファイルを正しく読み込めませんでした。撮影した動画の形式やファイルが壊れていないか確認してください。"
        elif "empty" in error_str.lower() or "index out of range" in error_str.lower():
            error_message = "動画から野球のフォーム（骨格）を検出できませんでした。全身が映るようにアングルや明るさを確認してください。"
        else:
            error_message = f"解析中にエラーが発生しました。理由: {error_str} (動画を短くしたり圧縮すると解決する場合があります)"

        error_html = f"""
        <div class="advice-item error-mode">
            <h3>⚠️ 解析エラー詳細報告</h3>
            <p><b>【動画の解析に失敗しました】</b></p><br><br>
            <p>原因: {error_message}</p>
        </div>
        """
        return jsonify({"weight_rate": "測定不能", "ai_data": error_html, "chart_data": []})

    # 一時ファイルの削除
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    # データ不足ガード
    if len(rates) < 3:
        error_html = """
        <div class="advice-item error-mode">
            <h3>⚠️ 解析エラー詳細報告</h3>
            <p><b>【動画の解析に失敗しました】</b></p><br><br>
            <p>原因: 動画から十分な長さの骨格データを検出できませんでした。全身が映るようにもう一度撮影してください。</p>
        </div>
        """
        return jsonify({"weight_rate": "測定不能", "ai_data": error_html, "chart_data": []})

    # 体重移動データの算出
    start_pos = rates[0]
    max_pos = max(rates)
    min_pos = min(rates)
    end_pos = rates[-1]
    
    # 💡 85%固定の奇跡を解消：動画の中で「一番大きくピッチャー側に踏み込んだ瞬間（最大値）」を画面のメイン数字にします！
    display_rate = max_pos

    # Geminiによるアドバイス生成
    ai_advice = "<p>AIアドバイスの生成に失敗しました。</p>"
    if GOOGLE_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"""
            プロの野球バッティングコーチとして、以下のスイングデータ（体重移動の推移）を分析し、熱血かつ具体的な指導レポートを作成してください。

            【スイングデータ（数値）】
            ・構え〜始動時の位置: {start_pos:.1f}%
            ・スイング中の最大移動: {max_pos:.1f}%
            ・スイング中の最小移動: {min_pos:.1f}%
            ・フォロースルー時の位置: {end_pos:.1f}%

            ※0%に近いほどキャッチャー寄り（軸足）、100%に近いほどピッチャー寄り（前足）に重心があります。

            【レポートに必ず含める内容】
            1. 今回のスイングの「ここが良い！」というポイント（褒める）
            2. 重心移動から読み取れる課題点（突っ込み気味、または軸足に残りすぎ、など）
            3. 明日からの練習で意識すべき具体的な改善アドバイス

            丁寧なHTMLタグ（<h3>や<p>、<ul><li>など）を使って、スタイリッシュに読みやすく出力してください。
            """
            response = model.generate_content(prompt)
            if response.text:
                ai_advice = response.text
        except Exception as ai_err:
            print(f"Gemini Error: {ai_err}")
            ai_advice = f"<p>AI解析レポートの生成中にエラーが発生しました。({str(ai_err)})</p>"

    # グラフ用データ
    chart_labels = [f"{i+1}" for i in range(len(rates))]
    chart_data = {
        "labels": chart_labels,
        "values": rates
    }

    return jsonify({
        "weight_rate": f"{int(display_rate)}",
        "ai_data": ai_advice,
        "chart_data": chart_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=10000)
