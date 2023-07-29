import numpy as np
import cv2

img_color = cv2.imread('apple.png')
height, width = img_color.shape[:2]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

lower_green1 = (50, 150, 15)
upper_green1 = (80, 180, 255)
lower_green2 = (50, 181, 15)
upper_green2 = (80, 220, 255)
lower_green3 = (50, 221, 15)
upper_green3 = (80, 255, 255)


lower_red1 = (170, 70, 50)
upper_red1 = (180, 131, 255)
lower_red2 = (0, 70, 50)
upper_red2 = (4, 131, 255)
lower_red12 = (170, 132, 50)
upper_red12 = (180, 192, 255)
lower_red22 = (0, 132, 50)
upper_red22 = (4, 192, 255)
lower_red13 = (170, 193, 50)
upper_red13 = (180, 255, 255)
lower_red23 = (0, 193, 50)
upper_red23 = (4, 255, 255)


lower_blue1 = (120-10, 70, 30)
upper_blue1 = (120+10, 131, 255)
lower_blue2 = (120-10, 132, 30)
upper_blue2 = (120+10, 192, 255)
lower_blue3 = (120-10, 193, 30)
upper_blue3 = (120+10, 255, 255)

# 각 패턴에 해당하는 색상 배열을 추가합니다.
patterns = [
    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 빨간색1

    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0],
               [0, 1, 1, 1, 1, 0],
               [0, 1, 1, 1, 1, 0]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 빨간색2
     
    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 빨간색3

    (np.array([[0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 초록색1

    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 0, 0, 0]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 초록색2
    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 초록색3

    (np.array([[1, 1, 1, 1, 1, 1],
               [1, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 1],
               [1, 1, 0, 0, 1, 1],
               [1, 1, 1, 1, 1, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 파란색1

    (np.array([[1, 1, 0, 0, 1, 1],
               [1, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 1],
               [1, 1, 0, 0, 1, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8)),  # 파란색2

    (np.array([[1, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
     np.array([0, 0, 0], dtype=np.uint8))  # 파란색3
]

while True:
    n = int(input("1. 빨간색, 2. 초록색, 3. 파란색: "))

    if n == 1:
        img_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        final_mask1 = cv2.bitwise_or(img_mask1, img_mask2)
        pattern1, base_color1 = patterns[3*n-3]
        img_mask12 = cv2.inRange(img_hsv, lower_red12, upper_red12)
        img_mask22 = cv2.inRange(img_hsv, lower_red22, upper_red22)
        final_mask2 = cv2.bitwise_or(img_mask12, img_mask22)
        pattern2, base_color2 = patterns[3*n-2]
        img_mask13 = cv2.inRange(img_hsv, lower_red13, upper_red13)
        img_mask23 = cv2.inRange(img_hsv, lower_red23, upper_red23)
        final_mask3 = cv2.bitwise_or(img_mask13, img_mask23)
        pattern3, base_color3 = patterns[3*n-1]
        final_mask=cv2.bitwise_or(final_mask1,cv2.bitwise_or(final_mask2,final_mask3))

        break
    elif n == 2:
        final_mask1 = cv2.inRange(img_hsv, lower_green1, upper_green1)
        pattern1, base_color1 = patterns[3*n-3]
        final_mask2 = cv2.inRange(img_hsv, lower_green2, upper_green2)
        pattern2, base_color2 = patterns[3*n-2]        
        final_mask3 = cv2.inRange(img_hsv, lower_green3, upper_green3)
        pattern3, base_color3 = patterns[3*n-1]      
        final_mask=cv2.bitwise_or(final_mask1,cv2.bitwise_or(final_mask2,final_mask3))
        break
    elif n == 3:
        final_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
        pattern1, base_color1 = patterns[3*n-3]
        final_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
        pattern2, base_color2 = patterns[3*n-2]        
        final_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
        pattern3, base_color3 = patterns[3*n-1]      
        final_mask=cv2.bitwise_or(final_mask1,cv2.bitwise_or(final_mask2,final_mask3))
        break
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")


img_result1 = img_color.copy()
img_result2 = img_color.copy()
img_result3 = img_color.copy()

# 패턴 입히기
pattern_size1 = pattern1.shape[0]  # 패턴의 크기
for y in range(0, height-pattern_size1+1, pattern_size1):
    for x in range(0, width-pattern_size1+1, pattern_size1):
        if np.any(final_mask1[y:y + pattern_size1, x:x + pattern_size1]):
            pattern_mask = pattern1 > 0
            inv_alpha = pattern1.astype(float) / 255.0

            # 명도 정보를 가져와서 패턴 색을 변경합니다.
            v_value = img_hsv[y:y + pattern_size1, x:x + pattern_size1, 2]
            v_value_inverted = 255 - v_value
            new_color = base_color1 * v_value_inverted[..., np.newaxis] / 255.0 + (255 - base_color1) * (1 - v_value_inverted[..., np.newaxis] / 255.0)

            # 새로운 색을 적용합니다.
            img_result1[y:y + pattern_size1, x:x + pattern_size1][pattern_mask] = new_color[pattern_mask].astype(np.uint8)

pattern_size2 = pattern2.shape[0]  # 패턴의 크기
for y in range(0, height-pattern_size2+1, pattern_size2):
    for x in range(0, width-pattern_size2+1, pattern_size2):
        if np.any(final_mask2[y:y + pattern_size2, x:x + pattern_size2]):
            pattern_mask = pattern2 > 0
            inv_alpha = pattern2.astype(float) / 255.0

            # 명도 정보를 가져와서 패턴 색을 변경합니다.
            v_value = img_hsv[y:y + pattern_size2, x:x + pattern_size2, 2]
            v_value_inverted = 255 - v_value
            new_color = base_color2 * v_value_inverted[..., np.newaxis] / 255.0 + (255 - base_color2) * (1 - v_value_inverted[..., np.newaxis] / 255.0)

            # 새로운 색을 적용합니다.
            img_result2[y:y + pattern_size2, x:x + pattern_size2][pattern_mask] = new_color[pattern_mask].astype(np.uint8)


pattern_size3 = pattern3.shape[0]  # 패턴의 크기
for y in range(0, height-pattern_size3+1, pattern_size3):
    for x in range(0, width-pattern_size3+1, pattern_size3):
        if np.any(final_mask3[y:y + pattern_size3, x:x + pattern_size2]):
            pattern_mask = pattern3 > 0
            inv_alpha = pattern3.astype(float) / 255.0

            # 명도 정보를 가져와서 패턴 색을 변경합니다.
            v_value = img_hsv[y:y + pattern_size3, x:x + pattern_size3, 2]
            v_value_inverted = 255 - v_value
            new_color = base_color3 * v_value_inverted[..., np.newaxis] / 255.0 + (255 - base_color3) * (1 - v_value_inverted[..., np.newaxis] / 255.0)

            # 새로운 색을 적용합니다.
            img_result3[y:y + pattern_size3, x:x + pattern_size3][pattern_mask] = new_color[pattern_mask].astype(np.uint8)

img_result = cv2.bitwise_or(img_result1, cv2.bitwise_or(img_result2,img_result3))



cv2.imshow('img_color', img_color)
cv2.imshow('img_mask', final_mask)
cv2.imshow('img_result', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()