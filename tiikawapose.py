import cv2
import mediapipe as mp
import numpy as np

# MediaPipeとOpenCVの設定
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 貼り付ける画像の読み込み
face_image = cv2.imread('character_face.png')  # 貼り付けたい画像パスを指定
face_image = cv2.resize(face_image, (100, 100))  # サイズ調整

# カメラ映像の取得
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("カメラからフレームを取得できませんでした。")
            break

        # 映像をRGBに変換
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # 再びBGRに変換して描画
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 骨格の描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 顔の位置を取得
            nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_x, nose_y = int(nose_landmark.x * frame.shape[1]), int(nose_landmark.y * frame.shape[0])

            # 顔画像を貼り付け
            overlay = image.copy()
            overlay_height, overlay_width = face_image.shape[:2]
            x1, y1 = nose_x - overlay_width // 2, nose_y - overlay_height // 2
            x2, y2 = x1 + overlay_width, y1 + overlay_height
            if 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0]:  # 範囲チェック
                alpha_face = face_image[:, :, 3] / 255.0 if face_image.shape[2] == 4 else np.ones((overlay_height, overlay_width))
                for c in range(0, 3):
                    image[y1:y2, x1:x2, c] = alpha_face * face_image[:, :, c] + (1 - alpha_face) * image[y1:y2, x1:x2, c]

        # 画像を表示
        cv2.imshow('Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
