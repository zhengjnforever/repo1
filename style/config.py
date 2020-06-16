#!/user/bin/env python
# -*- coding:utf-8 -*-
#配置数据库的文件
DIALECT ='mysql'
DRIVER = 'pymysql'
USERNAME = 'root'
PASSWORD = '' # 此处填写你的数据库密码,会用xmalp暂时还没设置密码
HOST = 'localhost' # 部署到服务器不能用127.0.0.1 得用localhost 对应服务器的主机地址
PORT = '3306'
DATABSE = 'clothesdata'# 此处为你建的数据库的名称
SQLALCHEMY_DATABASE_URI ="{}+{}://{}:{}@{}:{}/{}?charset=utf8".format(DIALECT,DRIVER,USERNAME,PASSWORD,HOST,PORT,DATABSE)
#指定配置，用来省略提交操作
SQLALCHEMY_TRACK_MODIFICATIONS = False