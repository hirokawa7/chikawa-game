import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# MediaPipeとOpenCVの設定
mp_pose = mp.solutions.pose

# 貼り付ける透過背景付き画像の読み込み
face_image = cv2.imread('./images/chi1/character_face.png', cv2.IMREAD_UNCHANGED)
left_eye_image = cv2.imread('./images/chi1/eye_image.png', cv2.IMREAD_UNCHANGED)
right_eye_image = cv2.imread('./images/chi1/eye_image.png', cv2.IMREAD_UNCHANGED)
mouth_image = cv2.imread('./images/chi1/mouth_image.png', cv2.IMREAD_UNCHANGED)

# カメラ映像の取得
cap = cv2.VideoCapture(0)

# 黒画像を作成
black_img0 = Image.new('RGBA', (640, 480), (0, 0, 0, 200))
black_img0.save('black_image_with_alpha.png')
black_img = cv2.imread('black_image_with_alpha.png', cv2.IMREAD_UNCHANGED)

# 画像をリサイズする関数
def resize_image(image, scale):
    # スケーリング処理
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image

# 画像を重ねる関数
def overlay_image(bg_img, fg_img, position, angle=0, scale=1.0):
    
    fg_img = resize_image(fg_img, scale)

    x, y = position
    h, w = bg_img.shape[:2]
    fg_h, fg_w = fg_img.shape[:2]

    # 貼り付ける範囲を背景画像のサイズに収まるように調整
    x1, x2 = max(x - fg_w // 2, 0), min(x + fg_w // 2, w)
    y1, y2 = max(y - fg_h // 2, 0), min(y + fg_h // 2, h)
    fg_x1, fg_y1 = max(0, -x + fg_w // 2), max(0, -y + fg_h // 2)
    fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)

    if x1 < x2 and y1 < y2:
        for c in range(0, 3):  # BGRチャンネル
            alpha = fg_img[fg_y1:fg_y2, fg_x1:fg_x2, 3] / 255.0
            bg_img[y1:y2, x1:x2, c] = (alpha * fg_img[fg_y1:fg_y2, fg_x1:fg_x2, c] +
                                        (1 - alpha) * bg_img[y1:y2, x1:x2, c])

    return bg_img

# 背景を透過させる関数
def trans_back(fname):
    img = Image.open(fname)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(fname, "PNG")

# スケルトンを描画する関数
def draw_thick_skeleton(image, landmarks, connections, thickness=20, color=(255, 255, 255)):
    height, width = image.shape[:2]
    for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]
        
        # 座標をピクセル値に変換
        start_x, start_y = int(start_landmark.x * width), int(start_landmark.y * height)
        end_x, end_y = int(end_landmark.x * width), int(end_landmark.y * height)
        
        # 線を描画
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

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

        # 全身骨格の極太描画
        if results.pose_landmarks:
            draw_thick_skeleton(image, results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS, thickness=20, color=(255, 255, 255))

            # 目と口のランドマークを取得し、回転とスケーリングを考慮
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
            mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]

            # 画像上のピクセル距離を計算
            eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2) * frame.shape[1]

            # 基準とする目の距離(基準とするピクセルを指定)
            scale = eye_distance / 100

            # 位置を取得
            nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])
            left_eye_x, left_eye_y = int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])
            right_eye_x, right_eye_y = int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])
            mouth_x, mouth_y = int(mouth.x * frame.shape[1]), int(mouth.y * frame.shape[0])

            # 顔画像の貼り付け
            image = overlay_image(image, face_image, (nose_x, nose_y), angle=0, scale=scale)

            # 目の位置と角度を計算して貼り付け
            eye_angle = np.degrees(np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
            image = overlay_image(image, left_eye_image, (left_eye_x, left_eye_y), angle=eye_angle, scale=scale)
            image = overlay_image(image, right_eye_image, (right_eye_x, right_eye_y), angle=eye_angle, scale=scale)

            # 口の位置と角度を計算して貼り付け
            image = overlay_image(image, mouth_image, (mouth_x, mouth_y), angle=eye_angle, scale=scale)

        # 画像を表示
        cv2.imshow('Pose Estimation', image)

        # 's'キーで画像を保存
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 1. 黒画像に骨格を描画
            skeleton_image = black_img.copy()
            draw_thick_skeleton(skeleton_image, results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS, thickness=20, color=(255, 255, 255))

            # 2. 画像を骨格の上に重ねる
            skeleton_image = overlay_image(skeleton_image, face_image, (nose_x, nose_y), angle=0, scale=scale)
            skeleton_image = overlay_image(skeleton_image, left_eye_image, (left_eye_x, left_eye_y), angle=eye_angle, scale=scale)
            skeleton_image = overlay_image(skeleton_image, right_eye_image, (right_eye_x, right_eye_y), angle=eye_angle, scale=scale)
            skeleton_image = overlay_image(skeleton_image, mouth_image, (mouth_x, mouth_y), angle=eye_angle, scale=scale)

            # 口の位置と角度を計算して貼り付け
            image = overlay_image(image, mouth_image, (mouth_x, mouth_y), angle=eye_angle, scale=scale)

            # 3. 背景透過処理を行い保存
            cv2.imwrite("captured_image.jpg", skeleton_image)
            im = Image.open('captured_image.jpg')
            im.save('captured_image.png')
            trans_back('captured_image.png')
            print("Image saved!")

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
