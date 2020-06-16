#!/user/bin/env python
# -*- coding:utf-8 -*-

from exts import db

class User(db.Model):
    #用户模型
    __tablename__='user'
    Uid=db.Column(db.Integer,primary_key=True,autoincrement=True)
    Openid=db.Column(db.String(255,'utf8_general_ci'))
    #用法：user=User(openid)
    def __init__(self,Openid):
        self.Openid=Openid
    def save(self):
        db.session.add(self)
        db.session.commit()

class Clothes(db.Model):
    #服装数据模型
    __tablename__='clothes'
    Cid=db.Column(db.Integer,primary_key=True,autoincrement=True)
    Uid=db.Column(db.Integer)
    Cpic=db.Column(db.String(255,'utf8_general_ci'))
    Cname=db.Column(db.String(255,'utf8_general_ci'))
    Cclass=db.Column(db.Integer)
    def __init__(self,Uid,Cpic,Cname,Cclass):
        self.Uid=Uid
        self.Cpic=Cpic
        self.Cname=Cname
        self.Cclass=Cclass
    def save(self):
        db.session.add(self)
        db.session.commit()
