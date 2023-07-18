import numpy as np
import cv2

img_color = cv2.imread('apple.png')  # 이미지 파일을 컬러로 불러옴
height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환

# upgarde_green
lower_green = (50, 150, 15) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_green = (80, 255, 255) 


# red
lower_red1 = (170, 70, 50) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_red1 = (180, 255, 255)
lower_red2 = (0, 70, 50)
upper_red2 = (4, 255, 255)

# blue
lower_blue = (120-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (120+10, 255, 255)

patterns = [
    np.array([[1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
    np.array([[1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1]], dtype=np.uint8) * 255,  # 체크무늬 패턴
    np.kron(np.eye(4, dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)) * 50  # 대각선 패턴
]

while True:
    n = int(input("1. 빨간색, 2. 초록색, 3. 파란색: "))

    if n == 1 :
        img_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        final_mask = cv2.bitwise_or(img_mask1, img_mask2)
        pattern = patterns[n-1]
        break
    elif n == 2:
        final_mask = cv2.inRange(img_hsv, lower_green, upper_green) 
        pattern = patterns[n-1]
        break
    elif n == 3:
        final_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) 
        pattern = patterns[n-1]
        break
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")


img_result = img_color.copy()

# 패턴 입히기
pattern_size = pattern.shape[0]  # 패턴의 크기
for y in range(0, height-pattern_size+1, pattern_size):
    for x in range(0, width-pattern_size+1, pattern_size):
        if np.any(final_mask[y:y + pattern_size, x:x + pattern_size]):
            img_result[y:y + pattern_size, x:x + pattern_size][pattern > 0] = 150

cv2.imshow('img_color', img_color)
cv2.imshow('img_mask', final_mask)
cv2.imshow('img_result', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()