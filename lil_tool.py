import sys

f = open("tool.txt",'w')
s=""
for i in range(784):
    s = s + ",col" + str(i)

f.write("label"+s)