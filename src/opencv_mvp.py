import numpy as np
import cv2

img_color = cv2.imread('apple.png')  # 이미지 파일을 컬러로 불러옴
height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환

# 색상 영역과 패턴 종류를 정의
color_ranges = [
    ((60, 40, 40), (100, 255, 255)),  # 초록색 범위
    ((170, 50, 50), (180, 255, 255)),  # 빨간색 범위
    ((120 - 10, 30, 30), (120 + 10, 255, 255))  # 파란색 범위
]

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
    n = int(input("1. 초록색, 2. 빨간색, 3. 파란색: "))
    if 1 <= n <= 3:
        color_range = color_ranges[n-1]
        pattern = patterns[n-1]
        break
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")

lower_color = np.array(color_range[0])
upper_color = np.array(color_range[1])

final_mask = cv2.inRange(img_hsv, lower_color, upper_color)
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