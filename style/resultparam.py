
from fashion_config import base_to_style ,style_config




#服装分类
#获取对应服装款式并返回
def to_Type(baseid):
    #id_type为服装的分类id
    id_type = baseid
    #print(id_type)
    #arr_type存放服装某一分类对应弹出的标签（服装款式数组）
    arr_type = base_to_style.get(id_type)
    #print(len(arr_type))
    result = []
    data={
        'code':1
    }
    result.append(data)
    for i in arr_type:
        dit = {"style_id": i,
               "style_name": style_config[i]}
        result.append(dit)
        #result = np.append(result,[dit],axis = 0)
    return result
    #print(result[0].get("style_id"))

def to_hsv(arr,id):
    try:
        color_if_center = 1
        # 获取衣服颜色数量
        color_Num = arr['colornum']
        if color_Num == 1 :
            num_color = 0
        else: num_color = 1
        #print(num_color)
    except:
        print(801)
    # 获取主颜色
    try:
        color_main = arr['color'][0]
        r = color_main[0]
        g = color_main[1]
        b = color_main[2]
        # print(r,g,b)
        #判断中间色
        if r>0 and r<=60 and g>0 and g<=60 and b>0 and b<=60 :
            hue = 2
            color_if_center = 0
        if r>235 and r<=255 and g>235 and g<=255 and b>235 and b<=255 :
            hue = 3
            color_if_center = 0
        if r>128 and r<=192 and g>128 and g<=192 and b>128 and b<=192 :
            hue = 4
            color_if_center = 0
    except:
        print(802)
    try:
        #rgb转为hsv
        _r=r/255
        _g=g/255
        _b=b/255
        color_max = max(_r,_g,_b)
        color_min = min(_r,_g,_b)
        #求v
        v = color_max
        #print(v*255)
        # 判断亮度
        if v > 0.85 and v < 1:
            value = 0
        else:
            value = 1
    except:
        print(803)
    #print(value)
    #求s
    try:

        s = (color_max-color_min)/color_max
        #print(s*255)
        #判断深浅色
        if s>0.85 and s<1 :
            staturation = 0
        else: staturation = 1
    except:
        print(804)
    #print(staturation)
    #求h
    try:
        if r / 255 == color_max:
            h = (_g - _b) / (color_max - color_min) * 60
        if g / 255 == color_max:
            h = 120 + (_g - _b) / (color_max - color_min) * 60
        if b / 255 == color_max:
            h = 240 + (_g - _b) / (color_max - color_min) * 60
        if h < 0:
            h = h + 360
        if color_if_center :
            #print(h/2)
            #判断色调
            if h>91 and h<270 :
                hue = 1
            else: hue = 0
    except:
        print(805)
    #print(hue)
    H = h/2


    S = s*255
    V = v*255

    #判断颜色色调
    if V>=0 and V<=46:
        rgb = 0
    else:
        if S>=43 and S<=255:
            if H>=0 and H<=10:
                if S>=43 and S<=160:
                    rgb = 11
                else:
                    rgb = 3
            elif H>10 and H<=25:
                if S >= 43 and S <= 160:
                    rgb =13
                else:
                    rgb = 4
            elif H > 25 and H <= 35:
                if S >= 43 and S <= 160:
                    rgb = 12
                else:
                    rgb = 5
            elif H > 35 and H <= 77:
                if S>=43 and S<=160:
                    rgb = 10
                else:
                    rgb = 6
            elif H > 77 and H <= 99:
                if S>=43 and S<=150:
                    rgb = 10
                else:
                    rgb = 7
            elif H > 99 and H <= 124:
                if S>=43 and S<=150:
                    rgb = 14
                elif S >= 235 and S <= 255:
                    rgb = 17
                else:
                    rgb = 8
            elif H > 124 and H <= 155:
                if S >= 43 and S <= 150:
                    rgb = 15
                elif S >= 235 and S <=255 :
                    rgb = 16
                else:
                    rgb = 9
            elif H>155 and H<=180:
                rgb = 18
        elif S>=0 and S<43:
            if V>=46 and V<=220:
                rgb = 1
            elif V > 220 and V <= 255:
                rgb = 2
        else:
            rgb=0
    #print(rgb)

    #统一传出格式
    item={
        "hue" : hue,
        "staturation" : staturation,
        "value" : value,
        "num_color" : num_color,
        "style_id" : id,
        "rgb" : rgb
    }

    #print(item)
    return item