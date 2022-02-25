test_dict = {'a':1, 's':5, 'd':2, 'f':3}
# sort_dict = sorted(test_dict.keys())
print(test_dict)

# 按照key值排序
sort_dict = {}
for k in sorted(test_dict):
    sort_dict[k] = test_dict[k]
print(sort_dict)