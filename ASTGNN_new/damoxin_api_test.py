import requests
from time import time

prediction = "1" # 是否预测流量，0表示否，1表示是
question = "拥堵情况" # 问题
timestamp = "21:10" # 时间戳
location = "金湖东路-红莲街" # 地点

# 定义请求的URL
url = "http://0.0.0.0:7860/model_api" # 本地地址，需要与你的api代码的主机和端口一致

# 定义请求的数据，以JSON格式传递
data = {
"prediction": prediction,
"question": question,
"timestamp": timestamp,
"location": location
}

data1 = {
"question": "三农指的是什么" # 问题
}
print(data)
# 发送POST请求，并获取服务器的响应
r = requests.post(json=data, url=url)


# 打印服务器的响应内容
print(r)
