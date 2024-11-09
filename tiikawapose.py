import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import random
import math


# MediaPipeとOpenCVの設定
mp_pose = mp.solutions.pose

# カメラ映像の取得
cap = cv2.VideoCapture(0)

# 黒画像を作成
black_img0 = Image.new('RGBA', (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), (0, 0, 0, 200))
black_img0.save('black_image_with_alpha.png')
black_img = cv2.imread('black_image_with_alpha.png', cv2.IMREAD_UNCHANGED)

# 装飾用カウントとフラグ
deco_count = 0
deco_flag = False

# うさぎかどうかのフラグ
usagi_flag = False

# ランダムにフォルダを選択する関数
def load_images_with_names_from_random_subfolder(parent_folder):

    # 親フォルダ内のサブフォルダ一覧を取得
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    if not subfolders:
        print("サブフォルダが見つかりません")

    # サブフォルダをランダムに選択
    random_subfolder = random.choice(subfolders)
    print(f"選択されたサブフォルダ: {random_subfolder[9:]}")

    return random_subfolder[9:]

# サブフォルダをランダムに選択
random_subfolder_name = load_images_with_names_from_random_subfolder("./images")

# 貼り付ける透過背景付き画像の読み込み
face_image = cv2.imread('./images/' + str(random_subfolder_name) + '/character_face.png', cv2.IMREAD_UNCHANGED)
left_eye_image = cv2.imread('./images/' + str(random_subfolder_name) + '/eye_image.png', cv2.IMREAD_UNCHANGED)
right_eye_image = cv2.imread('./images/' + str(random_subfolder_name) + '/eye_image.png', cv2.IMREAD_UNCHANGED)
mouth_image = cv2.imread('./images/' + str(random_subfolder_name) + '/mouth_image.png', cv2.IMREAD_UNCHANGED)

# 画像をリサイズする関数
def resize_image(image, scale):

    # scaleが正の値か確認
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    # 新しいサイズを計算
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

    # new_sizeの各値が0でないことを確認
    if new_size[0] <= 0 or new_size[1] <= 0:
        raise ValueError("Calculated new size has zero or negative dimensions.")

    # 画像をリサイズ
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


# 距離を計算する関数
def calculate_distance(landmark1, landmark2, image_width, image_height):

    x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
    x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 両手が顔の近くにあるかを判定する関数
def is_hands_near_face(landmarks, image_width, image_height, eye_distance, threshould_ratio=1.5):

    # 鼻と両手のランドマークを取得
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

    # 両手と顔の距離を計算
    left_distance = calculate_distance(nose, left_hand, image_width, image_height)
    right_distance = calculate_distance(nose, right_hand, image_width, image_height)

    # 両手と顔の距離が閾値いないなら、顔の近くにあると判定
    threshould_distance = threshould_ratio * eye_distance
    left_near_face = left_distance < threshould_distance
    right_near_face = right_distance < threshould_distance

    return None


# メインの処理
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
            eye_distance = calculate_distance(left_eye, right_eye, frame.shape[1], frame.shape[0])

            # 両手が顔の近くにあるか判定
            is_hands_near_face(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0], eye_distance)

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

        # ESCキーで終了
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
