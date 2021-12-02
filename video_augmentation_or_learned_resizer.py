#====================对于 video 数据的处理，比如数据增强，往往需要对单帧进行处理，最后再复原回来。=======================

#====================方法一：torch.stack（要求操作对象是元素为tensor的列表）===============================
# 打个栗子：video:(32,224,224,3) -> for 循环，对单帧 image:(224,224,3)进行数据增强。-> stack -> (32,new_H,new_W,3)
# 版面有限，下面以(4,2,2,3)举例
video = np.random.rand(4,2,2,3)
video_trfmed = []
for image_ in video:
    image_ = image_ + 1 # 替换成你的augmentation
    video_trfmed.append(torch.from_numpy(image_).float()) # 对tensor的.float()实现float64->float32
# stack恢复原Time维度
video_trfmed = torch.stack(video_trfmed)

#====================方法二：concat（实现过程麻烦，不推荐，效果同stack）===============================

#====================方法三：np.array（要求操作对象是元素为numpy数组的列表，如果操作对象和方法一一样...）===============================
video = np.random.rand(4,2,2,3)
video_trfmed = []
for image_ in video:
    image_ = image_ + 1 # 替换成你的augmentation
    video_trfmed.append(image_)
# np.array()恢复原Time维度
video_trfmed = np.array(video_trfmedd)

#===================方法四：利用for循环赋值（简单），仅适用于图片size不变的情况，不然会由于broadcasting产生bug================
video = np.random.rand(4,2,2,3)
for i in range(video.shape[0]):
  img_ = video[i] + 1 # 替换成你的augmentation
  video[i] = img_
return video
