import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import random
import math


# MediaPipeとOpenCVの設定
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# カメラ映像の取得
cap = cv2.VideoCapture(0)

# 黒画像を作成
black_img0 = Image.new('RGBA', (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (0, 0, 0, 200))
black_img0.save('black_image_with_alpha.png')
black_img = cv2.imread('black_image_with_alpha.png', cv2.IMREAD_UNCHANGED)

# うさぎかどうかのフラグ
usagi_flag = False

# はちかどうかのフラグ
hachi_flag = False

# キラキラエフェクトの半径と数
max_radius = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_particles = 30
sparkles = []

# ランダムな色と位置のキラキラエフェクトを生成
def initialize_sparkles(center_x, center_y):
    global sparkles
    sparkles = []

    for _ in range(num_particles):
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(10, max_radius)

        x = int(center_x + distance * np.cos(angle))
        y = int(center_y + distance * np.sin(angle))

        color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        
        sparkles.append({'x': x, 'y': y, 'color': color, 'radius': random.randint(10, 25)})

# キラキラエフェクトを描画
def draw_sparkles(image, center_x, center_y, scale=1.0):

    for sparkle in sparkles:
        x = int(center_x + (sparkle['x'] - center_x) * scale)
        y = int(center_y + (sparkle['y'] - center_y) * scale)

        cv2.circle(image, (x, y), sparkle['radius'], sparkle['color'], -1)

# 手が一定距離以上離れているかを判定する関数
def are_hands_spread_out(landmarks, image_width, threshold_ratio=0.7):
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
    
    # 左右の手のx座標の距離を計算
    hand_distance = abs(left_hand.x - right_hand.x) * image_width
    # 閾値を設定して、手が横に広がっているかどうか判定
    threshold_distance = image_width * threshold_ratio
    return hand_distance > threshold_distance

# 雨を描画するための関数
def draw_rainbow_rain_effect(image, num_drops=100, drop_length=20, drop_thickness=5):
    rainbow_rain_image = image.copy()
    height, width, _ = rainbow_rain_image.shape
    
    for _ in range(num_drops):
        # ランダムな色（虹色の範囲内）
        drop_color = (
            random.randint(0, 255),   # B
            random.randint(0, 255),   # G
            random.randint(0, 255)    # R
        )
        x = random.randint(0, width)
        y = random.randint(0, height - drop_length)
        end_y = y + drop_length
        cv2.line(rainbow_rain_image, (x, y), (x, end_y), drop_color, drop_thickness)
    
    # 透明度を設定
    alpha = 0.5
    cv2.addWeighted(rainbow_rain_image, alpha, image, 1 - alpha, 0, image)
    
    return image

# ランダムにフォルダを選択する関数
def load_images_with_names_from_random_subfolder(parent_folder):

    global usagi_flag, hachi_flag

    # 親フォルダ内のサブフォルダ一覧を取得
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    if not subfolders:
        print("サブフォルダが見つかりません")

    # サブフォルダをランダムに選択
    random_subfolder = random.choice(subfolders)
    print(f"選択されたサブフォルダ: {random_subfolder[9:]}")

    if 'usagi' in random_subfolder[9:]:
        usagi_flag = True

    if 'hachi3' in random_subfolder[9:]:
        hachi_flag = True

    return random_subfolder[9:]


# キャラクターのサブフォルダをランダムに選択
random_subfolder_name = load_images_with_names_from_random_subfolder("./images")

# 貼り付ける透過背景付き画像の読み込み
face_image = cv2.imread('./images/' + str(random_subfolder_name) + '/character_face.png', cv2.IMREAD_UNCHANGED)
left_eye_image = cv2.imread('./images/' + str(random_subfolder_name) + '/eye_image.png', cv2.IMREAD_UNCHANGED)
right_eye_image = cv2.imread('./images/' + str(random_subfolder_name) + '/eye_image.png', cv2.IMREAD_UNCHANGED)
mouth_image = cv2.imread('./images/' + str(random_subfolder_name) + '/mouth_image.png', cv2.IMREAD_UNCHANGED)

# 装飾のフォルダをランダムに選択
vr_subfolder_name = load_images_with_names_from_random_subfolder("./decora")
vr_image = cv2.imread('./decora/' + str(vr_subfolder_name) + '/vr.png', cv2.IMREAD_UNCHANGED)

# 涙目
sad_eye_image = cv2.imread('./options/sad_eye.png', cv2.IMREAD_UNCHANGED)

# 未来潜影の顔
mirai_hachi = cv2.imread('./options/miraiface.png', cv2.IMREAD_UNCHANGED)

# 太陽
sun_image = cv2.imread('./options/sun.png', cv2.IMREAD_UNCHANGED)

# 傘の顔
kasa_usagi = cv2.imread('./options/kasaface.png', cv2.IMREAD_UNCHANGED)

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

    img.resize((640, 360))

    img.save(fname, "PNG")

# スケルトンを描画する関数
def draw_thick_skeleton(image, landmarks, connections, thickness=20):

    global usagi_flag

    if usagi_flag == False:
        color = (255, 255, 255)
    else:
        color = (205, 235, 255)

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


# 顔の下で手を合わせているかどうかの関数
def is_hands_joined_below_face(landmarks, image_width, image_height, join_threshold=50, face_to_hand_ratio=0.5):

    # 顔（鼻）と手のランドマークを取得
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

    # 顔と両手の座標を取得
    nose_x, nose_y = int(nose.x * image_width), int(nose.y * image_height)
    left_hand_x, left_hand_y = int(left_hand.x * image_width), int(left_hand.y * image_height)
    right_hand_x, right_hand_y = int(right_hand.x * image_width), int(right_hand.y * image_height)

    # 両手が顔の下にあるか
    hands_below_face = left_hand_y > nose_y and right_hand_y > nose_y

    # 両手のx座標が近い（手が合わせているとみなす閾値以下）か
    hands_joined = abs(left_hand_x - right_hand_x) < join_threshold

    # 手が顔に近い位置にあるか（手のy座標が顔の少し下にあること）
    hands_close_to_face = (left_hand_y - nose_y) < (face_to_hand_ratio * image_height) and (right_hand_y - nose_y) < (face_to_hand_ratio * image_height)
    
    # すべての条件が満たされれば、顔の下で手を合わせていると判定
    if hands_below_face and hands_joined and hands_close_to_face:
        return True

    return False


# 片手が顔の近くにあるかを判定する関数
def is_hand_near_face(landmarks, image_width, image_height, eye_distance, threshould_ratio=2):

    hand_near_face = False

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

    if ((left_near_face == True) and (right_near_face == False)) or ((left_near_face == False) and (right_near_face == True)):
        hand_near_face = True

    return hand_near_face

# 両手が顔の近くにあるかを判定する関数
def is_hands_near_face(landmarks, image_width, image_height, eye_distance, threshould_ratio=5):

    hands_near_face = False

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

    if (left_near_face == True) and (right_near_face == True):
        hands_near_face = True

    return hands_near_face

# 右手が挙がっているかどうかを判定する関数
def is_right_hand_raised(landmarks, image_height):
    # 右手首、右肘、右肩のランドマークを取得
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # 右手が挙がっているかの判定条件
    wrist_above_elbow = (right_wrist.y * image_height) < (right_elbow.y * image_height)
    elbow_above_shoulder = (right_elbow.y * image_height) < (right_shoulder.y * image_height)

    # 両方の条件を満たしていれば、右手が挙がっていると判定
    return wrist_above_elbow and elbow_above_shoulder

# 右手が挙がっているかどうかを判定する関数
def is_left_hand_raised(landmarks, image_height):
    # 右手首、右肘、右肩のランドマークを取得
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

    # 右手が挙がっているかの判定条件
    wrist_above_elbow = (left_wrist.y * image_height) < (left_elbow.y * image_height)
    elbow_above_shoulder = (left_elbow.y * image_height) < (left_shoulder.y * image_height)

    # 両方の条件を満たしていれば、右手が挙がっていると判定
    return wrist_above_elbow and elbow_above_shoulder


# 画像保存用の番号
i=0

# メインループ内でのエフェクト描画
effect_active = False
effect_frame_count = 0
effect_duration = 30  # エフェクトの継続フレーム数

# メインの処理
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("カメラからフレームを取得できませんでした。")
            break

        # 映像をRGBに変換
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # ポーズ、手、顔のランドマークを検出
        results = pose.process(image)
        # hands_results = hands.process(image)
        # face_results = face_mesh.process(image)

        # 再びBGRに変換して描画
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 全身骨格の極太描画
        if results.pose_landmarks:
            draw_thick_skeleton(image, results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS, thickness=20)

            # 目と口のランドマークを取得し、回転とスケーリングを考慮
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
            mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]

            # 右手のランドマーク
            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

            # 画像上のピクセル距離を計算
            eye_distance = calculate_distance(left_eye, right_eye, frame.shape[1], frame.shape[0])

            # 両手が顔の近くにあるか判定
            hands_near_face = is_hands_near_face(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0], eye_distance)

            # 片手が顔の近くにあるか判定
            hand_near_face = is_hand_near_face(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0], eye_distance)

            # 右手が挙がっているか判定
            right_hand_raised = is_right_hand_raised(results.pose_landmarks.landmark, frame.shape[0])
            left_hand_raised = is_left_hand_raised(results.pose_landmarks.landmark, frame.shape[0])

            # 両手を合わせているか判定
            hands_joined = is_hands_joined_below_face(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

            # 両手があがっているときキラキラ
            if right_hand_raised and left_hand_raised and not effect_active:
                effect_active = True
                effect_frame_count = 0
                nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])
                initialize_sparkles(nose_x, nose_y)

            # 基準とする目の距離(基準とするピクセルを指定)
            scale = eye_distance / 100

            # 位置を取得
            nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])
            left_eye_x, left_eye_y = int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])
            right_eye_x, right_eye_y = int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])
            mouth_x, mouth_y = int(mouth.x * frame.shape[1]), int(mouth.y * frame.shape[0])
            right_hand_x, right_hand_y = int(right_hand.x * frame.shape[1]), int(right_hand.y * frame.shape[0])

            # エフェクト描画
            if effect_active:
                scale_ef = effect_frame_count / effect_duration  # エフェクトの広がり具合を計算
                draw_sparkles(image, nose_x, nose_y, scale_ef)
                effect_frame_count += 1
                if effect_frame_count >= effect_duration:
                    effect_active = False  # エフェクトを終了
            
            # 両手が横に広がっているかどうかを判定
            hands_spread = are_hands_spread_out(results.pose_landmarks.landmark, frame.shape[1])

            # 雨のエフェクト
            if hands_spread == True:
                image = draw_rainbow_rain_effect(image)

            # 顔画像の貼り付け
            if (right_hand_raised == True) and (hachi_flag == True):
                image = overlay_image(image, mirai_hachi, (nose_x, nose_y), angle=0, scale=scale)
            elif (hands_joined == True) and (usagi_flag == True):
                image = overlay_image(image, kasa_usagi, (nose_x, nose_y), angle=0, scale=scale)
            else:
                image = overlay_image(image, face_image, (nose_x, nose_y), angle=0, scale=scale)

            # 目の位置と角度を計算して貼り付け
            eye_angle = np.degrees(np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
            if hands_near_face == True and hand_near_face == False:
                image = overlay_image(image, sad_eye_image, (left_eye_x, left_eye_y), angle=eye_angle, scale=scale)
                image = overlay_image(image, sad_eye_image, (right_eye_x, right_eye_y), angle=eye_angle, scale=scale)
            else:
                image = overlay_image(image, left_eye_image, (left_eye_x, left_eye_y), angle=eye_angle, scale=scale)
                image = overlay_image(image, right_eye_image, (right_eye_x, right_eye_y), angle=eye_angle, scale=scale)

            # 口の位置と角度を計算して貼り付け
            image = overlay_image(image, mouth_image, (mouth_x, mouth_y), angle=eye_angle, scale=scale)

            # 片手が顔の近くにあるとき、装飾を貼り付け
            if hand_near_face == True:
                image = overlay_image(image, vr_image, (nose_x, nose_y), angle=0, scale=scale)

            # 右手を挙げているとき貼り付け
            if (right_hand_raised == True) and (left_hand_raised == False) and (hachi_flag == False):
                image = overlay_image(image, sun_image, (right_hand_x, right_hand_y), angle=0, scale=scale)

        # 画像を表示
        cv2.imshow('Pose Estimation', image)

        # 's'キーで画像を保存
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 1. 黒画像に骨格を描画
            skeleton_image = black_img.copy()
            draw_thick_skeleton(skeleton_image, results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS, thickness=20)

            # 2. 画像を骨格の上に重ねる
            if (right_hand_raised == True) and (hachi_flag == True):
                skeleton_image = overlay_image(skeleton_image, mirai_hachi, (nose_x, nose_y), angle=0, scale=scale)
            elif (hands_joined == True) and (usagi_flag == True):
                skeleton_image = overlay_image(skeleton_image, kasa_usagi, (nose_x, nose_y), angle=0, scale=scale)
            else:
                skeleton_image = overlay_image(skeleton_image, face_image, (nose_x, nose_y), angle=0, scale=scale)

            # 目の位置と角度を計算して貼り付け
            if hands_near_face == True and hand_near_face == False:
                skeleton_image = overlay_image(skeleton_image, sad_eye_image, (left_eye_x, left_eye_y), angle=eye_angle, scale=scale)
                skeleton_image = overlay_image(skeleton_image, sad_eye_image, (right_eye_x, right_eye_y), angle=eye_angle, scale=scale)
            else:
                skeleton_image = overlay_image(skeleton_image, left_eye_image, (left_eye_x, left_eye_y), angle=eye_angle, scale=scale)
                skeleton_image = overlay_image(skeleton_image, right_eye_image, (right_eye_x, right_eye_y), angle=eye_angle, scale=scale)

            skeleton_image = overlay_image(skeleton_image, mouth_image, (mouth_x, mouth_y), angle=eye_angle, scale=scale)

            if hand_near_face == True:
                skeleton_image = overlay_image(skeleton_image, vr_image, (nose_x, nose_y), angle=0, scale=scale)

            if (right_hand_raised == True) and (left_hand_raised == False) and (hachi_flag == False):
                skeleton_image = overlay_image(skeleton_image, sun_image, (right_hand_x, right_hand_y), angle=0, scale=scale)

            # 3. 背景透過処理を行い保存
            cv2.imwrite("captured_image.jpg", skeleton_image)
            im = Image.open('captured_image.jpg')
            im.save('./output/captured_image'+str(i)+'.png')
            trans_back('./output/captured_image'+str(i)+'.png')
            print("Image saved!")
            i = i+1

        # ESCキーで終了
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
