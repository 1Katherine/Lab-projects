# 找出 所有指标 mean_k8s大于mean_yarn的配置个数
import os_category_map
from mean_count_统计_code import micro_category_map

non_meaning = 1
count_map = {}
appear_count = {}
for name in ["50配置"]:
    for prefix in ["os", "micro"]:

        if prefix == "os":
            count_map = dict(
                zip(os_category_map.index_category.keys(),
                    [0 for i in range(len(os_category_map.index_category))]))  # 记录mean_k8s大于mean_yarn的个数
            count_map = dict(count_map)

            appear_count = dict(
                zip(os_category_map.index_category.keys(),
                    [0 for i in range(len(os_category_map.index_category))]))  # 记录该指标有没有出现过，1表示出现过，0是未出现

        else:
            count_map = dict(
                zip(micro_category_map.index_category.keys(),
                    [0 for i in range(len(micro_category_map.index_category))]))
            count_map = dict(count_map)

            appear_count = dict(
                zip(micro_category_map.index_category.keys(),
                    [0 for i in range(len(micro_category_map.index_category))]))

        for size in [50, 75, 100]:
            fileObj = open(
                "E:\ADE\compare_2\mean_count_统计_code\mean_count\\" + name + "\\" + prefix + "\\" + "mean_count_" + str(
                    size) + "g.txt")
            for line in fileObj:
                print(line)
                str_list = line.split(":")
                # print(str_list[0])
                # print(str_list[1])
                try:
                    count_map[str_list[0]] += int(str_list[1])
                    print(str_list[1])
                    appear_count[str_list[0]] = 1
                except Exception:
                    non_meaning += 1

        count_map = sorted(count_map.items(), key=lambda component: component[1], reverse=True)
        appear_count_dict = appear_count
        appear_count = sorted(appear_count.items(), key=lambda component: component[1], reverse=True)

        appear_file = open(prefix + "_appear_count.txt", mode='w')
        for line in appear_count:
            appear_file.write(line[0])
            appear_file.write(":")
            appear_file.write(str(line[1]))
            appear_file.write("\n")
        appear_file.close()

        count_file = open(prefix + "_total_count.txt", mode='w')
        for line in count_map:
            if appear_count_dict[line[0]] == 1:
                count_file.write(line[0])
                count_file.write(":")
                count_file.write(str(line[1]))
                count_file.write("\n")
        count_file.close()
