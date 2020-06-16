#!/user/bin/env python
# -*- coding:utf-8 -*-
from flask import request
from models import *
import json,datetime,random
from exts import db
import os
from resultparam import *

from detectedClothes import *
from score import *
import imageio
#测试成功
def to_Data():
    # data = request.get_data()  # 获取前端数据
    # data = str(data, 'utf-8')  # 转utf-8
    # data = json.loads(data)  # json转字典
    try:
        data = json.loads(request.get_data().decode("utf-8"))
    except:
        return {}
    if data:
        return data
    else:
        return {}

def to_Json(list = None):
    if list:
        data = json.dumps(list, ensure_ascii = False)
    else:
        data = "0"
    return data

#查找是否已经保存过这个用户测试成功
def selectnewuser(opid):
    result=User.query.filter(User.Openid == opid).first()
    if not result:
        return False
    else :
        return result.Uid

#添加用户测试成功
def Adduser(opid):
    nuser = User(Openid=opid)
    try:
        db.session.add(nuser)
        db.session.flush()
        newuid = nuser.Uid
        db.session.commit()
    except:
        db.session.rollback()
    #创建好用户对应的存衣服的文件夹
    Clothespic=os.path.join('static/users/'+str(newuid))
    os.makedirs(Clothespic)
    #创建存用户服装数据文件
    jsontext={}
    Clothesdata='static/users/'+str(newuid)+'/data.json'
    with open(Clothesdata, 'w') as dump_f:
        json.dump(jsontext, dump_f)
    #创建临时存放用户数据的文件夹
    staticarr='static/users/'+str(newuid)+'/staticarr.json'
    with open(staticarr, 'w') as dump_f:
        json.dump(jsontext, dump_f)
    return newuid

#检查衣服款式
def checkstylefun(Uid,img,MODEL):
    datapartone = getdata(img,MODEL)
    num = datapartone['num']
    # 如果没有衣服，则返回code=0，说明没衣服
    if num < 1:
        result = []
        data = {
            'code': 0
        }
        result.append(data)
        backdata =result
        return backdata
    # 如果衣服数量大于1
    elif num > 1:
        result = []
        data = {
            'code': 2
        }
        result.append(data)
        backdata = result
        return backdata
    else:
        staticfile = 'static/users/' + str(Uid) + '/staticarr.json'
        staticarr = datapartone
        with open(staticfile, 'w') as dump_f:
            json.dump(staticarr, dump_f)

        # 获取衣服的数组
        clotheslist = datapartone['Clotheslist']
        # 获取衣服识别出来的类别ID号
        baseid = clotheslist[0]['classid']
        # 获取衣服id号和对应的服装名字
        stylelist = getstylelist(baseid)
        return stylelist  # 返回给宝桐前端

#获取衣服款式列表
def getstylelist(baseid):
    backdata=to_Type(baseid)
    return backdata



#获取衣橱里的衣服
def getclotheslist(Uid,classid):
    clotheslist=[]
    clothesinfo=Clothes.query.filter(Clothes.Uid==Uid,Clothes.Cclass==classid).all()
    for row in range(len(clothesinfo)):
        clothes={
            'Cpic':clothesinfo[row].Cpic,
            'Cname':clothesinfo[row].Cname
        }
        clotheslist.append(clothes)
    return clotheslist

#上传衣服
def uploadpicc(Uid,styleid):
    try:
        #获取临时文件数据
        staticfile = 'static/users/' + str(Uid) + '/staticarr.json'
        with open(staticfile, 'r') as load_f:
            staticarr = json.load(load_f)
    except:
        return "302"

    try:
        now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        rand_num = random.randint(10, 99)  # 随机10到99
        Cname = str(now_time) + str(rand_num)
        picname = Cname + '.jpg'  # 合成
        img = staticarr['Clotheslist'][0]['img']
        #保存图片到服务器
        Cpic='static/users/'+str(Uid)+'/'+picname
        imageio.imsave(Cpic,img)
    except:
        return "303"
    try:
        classid = staticarr['Clotheslist'][0]['classid']
        Cclass = 0
        if classid > 6:
            Cclass = 1

        # 保存衣服到数据库
        newclothes = Clothes(Uid=Uid, Cpic=Cpic, Cname=Cname,Cclass=Cclass)
        newclothes.save()
    except:
        return "304"

    # try:
        # 衣服的详细数据
    arr=staticarr['Clotheslist'][0]
    result=to_hsv(arr,styleid)

        #保存服装数据信息到json文件中去
    Cdatadir="static/users/"+str(Uid)+"/"+"data.json"
    with open(Cdatadir,'r') as load_f:
        load_dict=json.load(load_f)
    load_dict[str(Cname)]=result
    with open(Cdatadir,'w') as dump_f:
        json.dump(load_dict,dump_f)
    # except:
    #     return "305"

    #返回保存的信息
    backdata={
        'Cimgurl':Cpic,
        'Cname':Cname
    }
    return backdata

#在个人衣橱选了衣服后评分
def choosescore(Uid,cnamelist):
    Cdatadir = "static/users/" + str(Uid) + "/" + "data.json"
    with open(Cdatadir, 'r') as load_f:
        load_dict = json.load(load_f)
    param = {}
    param['upitem'] = load_dict[str(cnamelist[0])]
    param['downitem'] = load_dict[str(cnamelist[1])]
    # print(param)
    score = getscore(param)
    return score




#拍照获取款式表
def getphotostyle(Uid,img,MODEL):

    datapartone = getdata(img,MODEL)
    num = datapartone['num']

    # 如果没有衣服，则返回code=0，说明没衣服
    if num == 0:
        result = []
        data = {
            'code': 0
        }
        result.append(data)
        backdata = to_Json(result)
        return backdata
    # 如果衣服数量大于1
    elif num == 1:
        result = []
        data = {
            'code': 1
        }
        result.append(data)
        backdata = to_Json(result)
        return backdata
    elif num == 2:
        staticfile = 'static/users/' + str(Uid) + '/staticarr.json'
        staticarr = datapartone
        with open(staticfile, 'w') as dump_f:
            json.dump(staticarr, dump_f)
        clotheslist = datapartone['Clotheslist']
        backdata = []
        data = {
            'code': 2
        }
        backdata.append(data)
        for part in clotheslist:
            baseid = part['classid']
            # 获取衣服id号和对应的服装名字
            result = getstylelist(baseid)
            backdata.append(result)
        # if backdata[1][1]['style_id'] > 19:
        #     temp=backdata[1]
        #     backdata[1]=backdata[2]
        #     backdata[2]=temp
        return backdata


def getpicscore(Uid,stylelidlist):
    Cdatadir= 'static/users/' + str(Uid) + '/staticarr.json'
    with open(Cdatadir, 'r') as load_f:
        staticarr = json.load(load_f)
    item1 = staticarr['Clotheslist'][0]
    upitem = to_hsv(item1, stylelidlist[0])
    item2 = staticarr['Clotheslist'][1]
    downitem = to_hsv(item2, stylelidlist[1])
    param = {
        'upitem': upitem,
        'downitem': downitem
    }
    # print(param)
    score=getscore(param)
    return score

def getaipair(Uid):
    # 获取上下装的Cname

    cur1= Clothes.query.filter(Clothes.Uid == Uid, Clothes.Cclass == 0).all()
    if not cur1:
        return [],0

    upresult = []

    for row in range(len(cur1)):
        upresult.append(cur1[row].Cname)


    cur2 = Clothes.query.filter(Clothes.Uid == Uid, Clothes.Cclass == 1).all()
    if not cur2:
        return [],0
    downresult = []
    for roww in range(len(cur2)):
        downresult.append(cur2[roww].Cname)

    # 获取已存在的数据
    score = 0
    scorelist = []
    for i in upresult:
        for j in downresult:
            data = [i, j]
            temp = choosescore(Uid,data)
            if temp >= score:
                score = temp
                part = {
                    'up': i,
                    'down': j,
                    'score': temp
                }
                scorelist.append(part)
    # print(scorelist)
    ailist = []
    # 筛选分数最高的组合
    for sco in scorelist:
        if sco['score'] == score:
            ailist.append(sco)
    # print(ailist)
    resultlist = []
    for ai in ailist:
        code1= ai['up']
        cur3= Clothes.query.filter(Clothes.Uid == Uid, Clothes.Cname ==code1).first()
        backdata = {}
        backdata['uppic'] = cur3.Cpic
        code2= ai['down']
        cur4 = Clothes.query.filter(Clothes.Uid == Uid, Clothes.Cname ==code2).first()
        backdata['downpic'] = cur4.Cpic
        resultlist.append(backdata)
    return resultlist, score


def deleteclothes(Cname,Cpic,Uid):
    # 删除数据库信息
    try:
        result=Clothes.query.filter(Clothes.Uid==Uid,Clothes.Cname==Cname)[0]
        db.session.delete(result)
        db.session.commit()
    except:
        print('d1')
        return  False
    # 删除图片信息
    print(Cpic)
    os.remove(Cpic)


    try:
        Cdatadir = "static/users/" + str(Uid) + "/" + "data.json"
        with open(Cdatadir, 'r') as load_f:
            load_dict = json.load(load_f)
        load_dict.pop(str(Cname))
        with open(Cdatadir, 'w') as dump_f:
            json.dump(load_dict, dump_f)
    except:
        print('d3')
        return  False

    return True


def twocheckstyle(Uid,img):
    return getphotostyle(Uid,img)

def savetwo(Uid,stylelidlist):
    staticfile = 'static/users/' + str(Uid) + '/staticarr.json'
    with open(staticfile, 'r') as load_f:
        staticarr = json.load(load_f)
    Cdatadir = "static/users/" + str(Uid) + "/" + "data.json"
    with open(Cdatadir, 'r') as load_f:
        load_dict = json.load(load_f)
    backlist=[]
    for i in range(len(stylelidlist)):
        try:
            styleid = stylelidlist[i]
            now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            rand_num = random.randint(10, 99)  # 随机10到99
            Cname = str(now_time) + str(rand_num)
            picname = Cname + '.jpg'  # 合成
            img = staticarr['Clotheslist'][i]['img']
            # 保存图片
            Cpic='static/users/'+str(Uid)+'/'+picname
            imageio.imsave(Cpic, img)
        except:
            print(101)
        try:
            classid = staticarr['Clotheslist'][i]['classid']
            Cclass = 0
            if classid > 6:
                Cclass = 1
            # 保存衣服到数据库
            newclothes = Clothes(Uid=Uid, Cpic=Cpic, Cname=Cname, Cclass=Cclass)
            newclothes.save()

        except:
            print(102)

        # 获取衣服数据

        arr=staticarr['Clotheslist'][i]
        result = to_hsv(arr, styleid)
        # print(result)

            # 保存衣服数据
            # try:

        try:
            # 保存衣服数据
            # try:

            load_dict[str(Cname)] = result

        except:
            print(104)
        backdata = {
            'Cimgurl': Cpic,
            'Cname': Cname
        }
        backlist.append(backdata)
    with open(Cdatadir, 'w') as dump_f:
        json.dump(load_dict, dump_f)
    return backlist

def loadmodel():

    ROOT_DIR = os.path.abspath("./")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn.h5")

    # Configurations
    class InferenceConfig(DeepFashion2Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    # config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config, model_dir="./logs/")

    # Select weights file to load 选择要加载的权重文件
    weights_path = WEIGHTS_PATH
    # weights_path = model.find_last()

    # Load weights 加载权重
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return model