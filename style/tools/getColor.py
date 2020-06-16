
import random as ran

from PIL import Image
from pylab import *

#######Get Color#######

# 加载图片，返回图片数据
def loadImage(im):
    #im = Image.open(path)  # 可以是许多不同的格式的照片
    pix = im.load()  # 获得图像的像素
    width = im.size[0]  # 获得图像的宽度
    height = im.size[1]  # 获得图像的高度
    data = width, height, pix, im  # 把这些width，height，pix，im这些值赋给data，后面KMeans方法里要用到这些值
    return data                   #返回图片数据

def getColor(image):
    k = 20  # 设置k均值初始点个数，也可设为10，结果不一样，如果改成10的话，show()图片的X坐标也要相应改变字母个数
    img = Image.fromarray(image.astype('uint8')).convert('RGB')
    data = loadImage(img)
    # 通过KMeans方法后返回的centroids，是k均值迭代后最终的中心点, count是这k个中心（类）的所包含的个数
    count, centroids = KMeans(k, data)
    for i in range(k):  # 因为有k个中心点
        h, s, v = centroids[i]
        r, g, b = Hsv2Rgb(h, s, v)
        centroids[i] = r, g, b

    im = data[3]  # im = Image.open(path)，就是得到图像对象,data[3]存的是图片
    colorNum,rgblist = show(im, count, centroids, k,img)  # 显示图像
    return rgblist

# 中心点k均值迭代  data为图片数据
def KMeans(k, data):
    width, height, pix,im = data  # 获得要处理图像的各个数据
    dataSet = [[0 for col in range(height)] for row in range(width)]  # 图像数据转化为hsv后的数据及其数据格式
    for x in range(width):
        for y in range(height):
            r, g, b = pix[x, y]  # 获取图像rgb值
            if r==g==b:
                continue
            hsv = h, s, v = rgb2hsv(r, g, b)  # 把rgb值转化为hsv值
            dataSet[x][y] = hsv    #储存hsv值序列
    dataSet = np.array(dataSet)  # 把dataSet数据转化为numpy的数组数据，以便待会获得初始点时，更好处理数据
    centroids = getCent(dataSet, k)  # 获得k个初始中心点

    # 循环迭代直到前一个centroids与当前centroids的根号距离满足一定条件
    while 1:
        count = [0 for i in range(k)]  # count用来统计各个中心类中的数据的个数
        myList = [[] for i in range(width * height)]  # mylist用来存放各个中心类中的数据
        preC = centroids  # preC保存前一个centroids的数据
        # 给每个点归类：判断各个像素属于哪个中心类，然后把hsv值放到所属类
        for x in range(width):
            for y in range(height):
                r, g, b = pix[x, y]  #根据每个像素点获取其rgb值
                if r == g == b:
                    continue
                hsv = h, s, v = rgb2hsv(r, g, b)  #rgb值转换成hsv值
                i = distEclud(hsv, centroids, k)  # 计算欧氏距离，获得该像素，也就是hsv所属中心类 ，返回该类下标
                myList[i].append((h, s, v))  # 把hsv值加到所属中心类
                count[i] += 1  # 相应所属类的个数增加

        # 一次所有点类别划分后，重新计算中心点 ，每个像素点的RGB值归类后，由该中心点计算均值取代该中心点的RGB值（HSV值）
        for i in range(k):
            size = len(myList[i])  # 各个类中的点个数
            sumh = sums = sumv = 0.0
            if (size == 0):
                continue
            else:
                for j in range(size):#获取该类中点的h,s,v值的总值
                    h, s, v = myList[i][j]
                    sumh += h
                    sums += s
                    sumv += v
            centroids[i] = sumh / size, sums / size, sumv / size  # 总值除以个数=取该类hsv分量的平均值

        norm = getDist(preC, centroids)  # 获得前一个centroids与当前centroids的根号距离
        if norm < 0.1:  # 距离小于0.1，则跳出循环
            break
#    for x in range (k):
#        print(x,"注意看：",myList[x]) ,myList[]储存的是各个类的 各个点 hsv数据序列
    return count, centroids  # 返回count：各个中心点数据的个数；centroids：最终迭代后的中心点

# hsv空间两点间欧氏距离，选出距离最小的类
def distEclud(hsv, centroids, k):
    h, s, v = hsv  # 获取当前像素的h，s，v值
    min = -1  # 用作判断centroids[i]是否为第一个中心点
    # 逐个计算当前hsv与各个类中心点的欧式距离，选出距离最小的类，有k个类
    for i in range(k):
        h1, s1, v1 = centroids[i]     #获得每个类中心点的h,s,v值
        minc = math.sqrt(math.pow(math.fabs(h - h1), 2) + math.pow(math.fabs(s - s1), 2) + math.pow(math.fabs(v - v1), 2))
        # minc = math.sqrt(math.pow(s*math.cos(h) - s1*math.cos(h1), 2) + math.pow(s*math.sin(h) - s1*math.sin(h1), 2) + \
        #     + math.pow(v - v1, 2))/math.sqrt(5)     # 欧氏距离计算公式
        # 用j表示当前hsv值属于第j个centroids
        if (min == -1):
            min = minc
            j = 0
            continue
        if (minc < min):
            min = minc
            j = i
    return j         #返回该空间点hsv距离最小的一个类 的下标

# 随机生成初始的质心（ng的课说的初始方式是随机选K个点），选择图像中最小的值加上随机值来生成
def getCent(dataSet, k):
    centroids = zeros((k, 3))  # 种子，k表示生成几个初始中心点，3表示hsv三个分量
    width, height = dataSet.shape  # 获得数据的长宽
    # 循环获得dataSet所有数据里面最小和最大的h，s，v值
    maxh, maxs, maxv = minh, mins, minv = 180, 0.5, 0.5  # 选择中值进行初始化
    for i in range(width):
        for j in range(height):
            if not dataSet[i][j]:
                continue
            h, s, v = dataSet[i][j]
            if h > maxh:
                maxh = h
            elif s > maxs:
                maxs = s
            elif v > maxv:
                maxv = v
            elif h < minh:
                minh = h
            elif s < mins:
                mins = s
            elif v < minv:
                minv = v
    rangeh = maxh - minh  # 最大和最小h值之差
    ranges = maxs - mins
    rangev = maxv - minv
    sum1 = 0
    sum2 = 0
    sum3 = 0
    # 生成k个初始点，hsv各个分量的最小值加上range的随机值
    #随机数生成的次数可控制精度，次数越多越稳定，但能识别出来的主颜色越少
    for i in range(k):
        for j in range(4):
            sum1 = minh + rangeh * ran.random()
            sum2 = mins + ranges * ran.random()
            sum3 = minv + rangev * ran.random()
        sum1 /= 4
        sum2 /= 4
        sum3 /= 4
        centroids[i] = sum1,sum2,sum3
    return centroids

# 前一个centroids与当前centroids的根号平方差，类中的每一个点与另一个类的每个点的距离
def getDist(preC, centroids):
    k, n = preC.shape  # k表示centroids的k个中心点（类中心点），n表示例如centroid[0]当中的三个hsv分量
    sum = 0.0  # 总距离
    for i in range(k):
        h, s, v = preC[i]
        h1, s1, v1 = centroids[i]
        distance = math.pow(math.fabs(h - h1), 2) + math.pow(math.fabs(s - s1), 2) + math.pow(math.fabs(v - v1), 2)
        sum += distance
    return math.sqrt(sum)    #返回总距离


#显示图片结果
def show(im, count, centroids, k,img):
    colorNum = 0
    rgblist = []

    x = k  # x坐标  通过对txt里的行数进行整数分解
    # 冒泡算法从大到小排序
    for i in range(k):
        max = count[i]
        m = i
        for j in range(i, k):
            if count[j] > max:
                max = count[j]
                m = j
        if i != m:  # 排序交换
            midcount = count[i]
            count[i] = count[m]
            count[m] = midcount
            mid = centroids[i]
            centroids[i] = centroids[m]
            centroids[m] = mid
    if x > 6:  # 取前6个中心类个数最大的颜色
        x = 6
    count_remove = 0  # 用于统计，剔除中心类中，类聚集的数据数小于5%的
    sum_count = float(sum(count))  # sum_count为总的数据数个数，也就是各个类聚集的总个数
    # 剔除中心类中，类聚集的数据数小于5%的
    for i in range(x):
        if count[x - i - 1] / sum_count < 0.125:  # 类聚集的数据数小于15%的，去除
            count_remove += 1
    x = x - count_remove  # k个类中，除去数据数小于15%的
    if x == 0:
        x = 1  # 确保有一个主颜色
    colorNum = x

    # 前6个中心类个数最大的颜色
    for i in range(0, x):
        temp = []
        rgb = centroids[i]
        temp.append(int(rgb[0]))
        temp.append(int(rgb[1]))
        temp.append(int(rgb[2]))
        rgblist.append(temp)

    return colorNum,rgblist


###tools###
def Hsv2Rgb(H, S, V):
    H /= 60.0  # sector 0 to 5
    i = math.floor(H)
    f = H - i  # factorial part of h
    p = V * (1 - S)
    q = V * (1 - S * f)
    t = V * (1 - S * (1 - f))
    if i == 0:
        R = V
        G = t
        B = p
    elif i == 1:
        R = q
        G = V
        B = p
    elif i == 2:
        R = p
        G = V
        B = t
    elif i == 3:
        R = p
        G = q
        B = V
    elif i == 4:
        R = t
        G = p
        B = V
    else:
        R = V
        G = p
        B = q
    return R * 255, G * 255, B * 255

def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r and g >= b:
        h = 60 * ((g - b) / df) + 0
    elif mx == r and g < b:
        h = 60 * ((g - b) / df) + 360
    elif mx == g:
        h = 60 * ((b - r) / df) + 120
    elif mx == b:
        h = 60 * ((r - g) / df) + 240
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v

def rgb2hsv2(R, G, B):
    mx = max(R, G, B)
    mn = min(R, G, B)
    if R == mx:
        H = (G - B) / (mx - mn)
    elif G == mx:
        H = 2 + (B - R) / (mx - mn)
    elif B == mx:
        H = 4 + (R - G) / (mx - mn)
    H = H * 60
    if H < 0:
        H = H + 360
    V = mx
    S = (mx - mn) / mx
    return H, S, V
