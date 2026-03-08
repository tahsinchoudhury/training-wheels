a = [
    [1, 2, 3],
    [4, 5, 6],
]

temp = a
index = (1, 0)
for i in index:
    print(i)
    temp = temp[i]

print(a[1][0])
print(temp)