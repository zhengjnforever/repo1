# -*- coding:utf-8 -*-
from io import BytesIO
import matplotlib.pyplot as plt
from flask import Flask,request
import requests
from flask import render_template
import config
from flask_cors import *
from methods import *
from models import *
import json


# 定义系统路径的变量
BASE_DIR = os.path.dirname(__file__)
# 定义静态文件的路径
static_dir = os.path.join(BASE_DIR, 'static')
# 定义模板文件的路径
templates_dir = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__)
CORS(app,supports_credentials=True) #解决跨域问题

app.config.from_object(config)   #对默认配置的修改和添加
db.init_app(app)

global MODEL
MODEL=loadmodel()


#主页
@app.route('/')
def hello_world():
    return render_template("home.html")

#用户注册1
#用户注册成功后返回Uid给用户
@app.route('/adduser',methods=['POST'])
def adduser():
    try:
        # 获取前端数据
        data = to_Data()
        # 开发者的appid
        appID = 'wxab1dba81bafd11f8'
        # 开发者的开发密码
        appSecret = 'd077a012fcd5113c0f1babef6f93c01f'
        # 用户的code
        code = data['code']
    except:
        #缺少参数
        return "101"

    # # 微信的服务器，到微信服务器获取openid
    try:
        # 2.向微信服务器发送http请求，获取access_token
        req_params = {
            'appid': appID,
            'secret': appSecret,
            'js_code': code,
            'grant_type': 'authorization_code'
        }
        wx_login_api = 'https://api.weixin.qq.com/sns/jscode2session'
        response_data = requests.get(wx_login_api, params=req_params)  # 向API发起GET请求
        data = response_data.json()
        openid_ = data['openid']  # 得到用户关于当前小程序的OpenId
        # print(openid_)
        # 判断这位用户在小程序保存过账号
    except:
        return "102"
    # openid_=data['openid']

    # try:
        #查看用户是否注册过
    Uid = selectnewuser(openid_)
        # print(Uid)
        #如果没注册就注册
    if not Uid:
            # 把新用户添加数据库里
        Uid = Adduser(openid_)
        #如果注册过就返回注册过的Uid
    # except:
    #     return "103"

    backdata = {
        'Uid': Uid
    }
    return json.dumps(backdata)

#上传衣服图片2 没问题
@app.route('/checkstyle',methods=['POST'])
def checkstyle():
    # try:
    file_obj=request.files.get('img')
    file_content=file_obj.read()
    img=plt.imread(BytesIO(file_content),"jpg")
    # print(type(img))
    Uid = request.form.get('Uid')
    # except:
    #     return "201"
    #对图片进行识别
    #昊天部分
    # print(img)
    stylelist=checkstylefun(Uid,img,MODEL)
    backdata = to_Json(stylelist)
    return backdata  # 返回给宝桐前端


#保存用户上传的衣服（单件）3 没问题
@app.route('/uploadpic',methods=['POST'])
def uploadpic():
    data=to_Data()
    try:
        Uid=data['Uid']
        styleid=data['styleid']
    except:
        return "302"

    backdata=uploadpicc(Uid,styleid)
    return json.dumps(backdata)


#获取用户已经上传的服装图4 无问题
@app.route('/getclothes',methods=['POST'])
def getclothes():
    data = to_Data()
    #选择衣服的类型（0：上装，1：下装）
    classid=data['code']
    Uid=data['Uid']
    clotheslist=getclotheslist(Uid,classid)
    data = to_Json(clotheslist)
    return data

#反馈用户评分5 无问题
@app.route('/toscore',methods=['POST'])
def toscore():
    data=to_Data()
    cnamelist=data['cnamelist']
    Uid=data['Uid']
    score=choosescore(Uid,cnamelist)

    return str(score)


#拍照评分1 6 伪问题
@app.route('/takephoto',methods=['POST'])
def takephoto():
    try:
        file_obj = request.files.get('img')
        file_content = file_obj.read()
        img = plt.imread(BytesIO(file_content), "jpg")
        Uid = request.form.get('Uid')
    except:
        return "601"
    result=getphotostyle(Uid,img,MODEL)
    data = to_Json(result)
    return data

#拍照评分2 7 无问题
@app.route('/takescore',methods=['POST'])
def takescore():
    data=to_Data()
    try:
        Uid = data['Uid']
        stylelidlist=data['stylelist']
    except:
        return "701"
    score=getpicscore(Uid, stylelidlist)
    return str(score)

#智能搭配 无问题
@app.route('/aipair',methods=['POST'])
def aipair():
    data=to_Data()
    Uid=data['Uid']
    result,score=getaipair(Uid)
    if len(result):
        backdata={
            'code':1,
            'score':score,
            'clotheslist':result
        }
    else:
        backdata={
            'code':0
        }
    return json.dumps(backdata)


@app.route('/deleteclothes',methods=['POST'])
def deleteClothes():
    data = to_Data()
    Cpic=data['Cpic']
    Cname=data['Cname']
    Uid=data['Uid']
    # print(Cpic,Cname,Uid)
    if deleteclothes(Cname,Cpic,Uid):
        backdata={
            'code':1
        }
        return json.dumps(backdata)
    else:
        backdata = {
            'code': 0
        }
        return json.dumps(backdata)

@app.route('/addtwoclothes',methods=['POST'])
def addtwo():
    data = to_Data()
    try:
        Uid = data['Uid']
        stylelidlist = data['stylelist']
    except:
        return "701"
    try:
        backlist=savetwo(Uid,stylelidlist)
        backdata={
            'code':1,
            "clotheslist":backlist
        }
    except:
        backdata={
            'code':0
        }
    return json.dumps(backdata)




if __name__ == '__main__':
    app.run(debug=True,threaded=True,host='0.0.0.0',port=5000,)
