#!/user/bin/env python
# -*- coding:utf-8 -*-

from fashion_config import add_score,cut_score,BASECOLOR,BASENUM,BASERGB,BASESTATURATION,BASESTYLE,BASEVALUE

# from tips import getresult_param
# 获取分数

def getscore(param):
# 初始化参数
#     param=getresult_param()
#     param={
#         "upitem":{
#             "hue":1,
#             "staturation":0,
#             "value":0,
#             "num_color":0,
#             "style_id":5
#         },
#         "downitem":{
#             "hue":1,
#             "staturation":0,
#             "value":0,
#             "num_color":0,
#             "style_id":24
#         }
#     }


    # 获取加减分
    basecolor=BASECOLOR
    basenum=BASENUM
    basestaturantion=BASESTATURATION
    basevalue=BASEVALUE
    basergb=BASERGB
    basestyle=BASESTYLE

    addcolor=add_score.get("color_score")
    addnum=add_score.get("num_score")
    addstaturantion=add_score.get("staturantion_score")
    addvalue=add_score.get("value_score")
    addrgb=add_score.get("rgb_score")
    addclothes=add_score.get("clothes_score")


    cutcolor = cut_score.get("color_score")
    cutnum=cut_score.get("num_score")
    cutstaturantion = cut_score.get("staturantion_score")
    cutvalue = cut_score.get("value_score")
    cutrgb = cut_score.get("rgb_score")
    cutclothes = cut_score.get("clothes_score")


    # 获取参数信息并附到变量上
    upitem=param.get("upitem")
    uphue=upitem.get("hue")
    upstaturation=upitem.get("staturation")
    upvalue=upitem.get("value")
    upnum_color=upitem.get("num_color")
    upstyle_id=upitem.get("style_id")
    up_rgb=upitem.get("rgb")

    downitem=param.get("downitem")
    downhue=downitem.get("hue")
    downstaturation=downitem.get("staturation")
    downvalue=downitem.get("value")
    downnum_color=downitem.get("num_color")
    downstyle_id=downitem.get("style_id")
    down_rgb=downitem.get("rgb")


    # 加分
    add_color=add_score.get("color")
    addcolorlist=add_color.get(uphue)
    if addcolorlist:
        for i in addcolorlist:
            if downhue==i:
                basecolor += addcolor
                break

    addnum_color=add_score.get("num_color")
    addnum_colorlist=addnum_color.get(upnum_color)
    if addnum_colorlist:
        for i in addnum_colorlist:
            if downnum_color==i:
                basenum+=addnum
                break

    addstaturation=add_score.get("staturation")
    addstaturationlist=addstaturation.get(upstaturation)
    if addstaturationlist:
        for i in addstaturationlist:
            if downstaturation==i:
                basenum+=addnum
                break


    add_vlaue = add_score.get("value")
    addvlauelist = add_vlaue.get(upvalue)
    if addvlauelist:
        for i in addvlauelist:
            if downvalue == i:
                basevalue += addvalue

    add_clothes=add_score.get("clothes")
    addclotheslist=add_clothes.get(upstyle_id)
    if addclotheslist:
        for i in addclotheslist:
            if downstyle_id==i:
                basestyle += addclothes
                break

    add_rgb=add_score.get("rgb")
    addrgblist=add_rgb.get(up_rgb)
    if addrgblist:
        for i in addrgblist:
            if down_rgb==i:
                basergb+=addrgb
                break

    # 减分
    cut_color = cut_score.get("color")
    cutcolorlist = cut_color.get(uphue)
    if cutcolorlist:
        for i in cutcolorlist:
            if downhue == i:
                basecolor += cutcolor
                break

    cutnum_color = cut_score.get("num_color")
    cutnum_colorlist = cutnum_color.get(upnum_color)
    if cutnum_colorlist:
        for i in cutnum_colorlist:
            if downnum_color == i:
                basenum += cutnum
                break

    cut_staturation = cut_score.get("staturation")
    cutstaturationlist = cut_staturation.get(upstaturation)
    if cutstaturationlist:
        for i in cutstaturationlist:
            if downstaturation == i:
                basestaturantion += cutstaturantion
                break

    cut_vlaue = cut_score.get("value")
    cutvlauelist = cut_vlaue.get(upvalue)
    if cutvlauelist:
        for i in cutvlauelist:
            if downvalue == i:
                basevalue += cutvalue

    cut_clothes = cut_score.get("clothes")
    cutclotheslist = cut_clothes.get(upstyle_id)
    if cutclotheslist:
        for i in cutclotheslist:
            if downstyle_id==i:
                basestyle += cutclothes
                break


    cut_rgb = cut_score.get("rgb")
    cutrgblist = cut_rgb.get(down_rgb)
    if cutrgblist:
        for i in cutrgblist:
            if down_rgb == i:
                basergb += cutrgb
                break


    # print(colorscore,stylescore)
    final_score=basecolor+basenum+basestaturantion+basevalue+basergb+basestyle
    # print(final_score)
    return final_score

