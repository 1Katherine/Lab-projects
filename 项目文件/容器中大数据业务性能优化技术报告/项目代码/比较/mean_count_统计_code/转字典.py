file_1= open("micro_category_map_1.txt",mode="r")
file= open("micro_category_map.py", mode="a+")

for line in file_1:
    print(line)
    line=line.strip("\n")
    file.write("\"" +line +"\""+":1,")

    file.write("\n")

