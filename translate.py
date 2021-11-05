# -*- coding: utf8
import http.client
import hashlib
import urllib.parse
import random
import json


def translate(q="This is a good day", fromLang="en", toLang="zh"):
    appid = ''
    secretKey = ''

    httpClient = None
    myurl = '/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    sign = appid+q+str(salt)+secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode())
    sign = m1.hexdigest()
    myurl = myurl+'?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
    result = ""

    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)
    #response是HTTPResponse对象
    response = httpClient.getresponse()
    result = response.read()

    temp = json.loads(result)
    result = temp['trans_result'][0]['dst']
    return result

if __name__ == '__main__':
    res = translate('This is a good day', fromLang="en", toLang="zh")
    # res = translate(u'今天是个好日子', fromLang="zh", toLang="en")
    print(res)