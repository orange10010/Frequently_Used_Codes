import time

#获取日期，格式化yyyy-mm-dd hh:mm:ss
#第一种方式
strtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
print(strtime)
