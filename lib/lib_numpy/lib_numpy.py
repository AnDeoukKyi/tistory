import numpy as np

# 생성
print("---생성---")
a = np.zeros((10,) + (5,) + (4,) + (3,), dtype=np.uint8)
print("현재차원\n", a.shape)

# 차원변경
print("---차원변경---")
b = np.zeros((20,))
print("b(20,)\n", b)
b = b.reshape(5, 4)
print("b(5, 4)\n", b)
b = b.reshape(2, 10)
print("b(2, 10)\n", b)

# 데이터 타입 변환
print("---데이터 타입 변환---")
c = np.zeros((10,))
print("현재타입\n", c.dtype)
c = c.astype(np.uint8)
print("현재타입\n", c.dtype)

# 스칼라곱
print("---스칼라곱---")
a1 = np.array([1, 2, 3])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
print("a1\n", a1)
print("a2\n", a2)
print("스칼라곱 결과 a1 * a2\n", a1 * a2)  # [1*1+2*2+3*3, 1*2+2*5, 1*3+2*6]
# 스칼라곱 차원오류
# print("---스칼라곱 차원오류---")
# a1 = np.array([1,2,])
# a2 = np.array([[1,2,3],[4,5,6]])
# print("a1\n" ,a1)
# print("a2\n" ,a2)
# print("스칼라곱 결과 a1 * a2\n", a1*a2)

# 내적
print("---내적---")
a1 = np.array([1, 2])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
print("a1\n", a1)
print("a2\n", a2)
print("dot연산 a1 dot a2\n", np.dot(a1, a2))  # [1*1+2*4, 1*2+2*5, 1*3+2*6]

# 합
print("---합---")
d = np.array([[1, 2, 3], [4, 5, 6]])
print("d\n", d)
print("axis없을때\n", np.sum(d))
print("axis=0일때\n", np.sum(d, axis=0))
print("axis=1일때\n", np.sum(d, axis=1))
