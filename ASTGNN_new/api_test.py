import requests
from time import time

timestamp='21:10'
location='金湖东路-红莲街'

url='http://localhost:1125/TrafficFlowForcast'
data={
    'timestamp':timestamp,
    'location':location
}
s=time()
r=requests.post(json=data,url=url).json()
e=time()
print(r['result'])
split_list = r['result'].split(";")
sum=0.0
for sl in split_list:
    print(sl)
    sum =sum+float(sl)
sum=sum/12
print(sum)


def average_speed(L, N):
    # 使用车辆的实际平均速度的公式计算速度
    v = L/N*12
    return v

L = 500 # 路段的长度，单位为米
N = sum # 路段上的车辆数，单位为辆

v = average_speed(L, N)
print(f"车辆的预测的平均速度是{v:.2f}公里/小时")


if v>40:
    res="通畅"
elif v>30:
    res="比较通畅"
elif v>20:
    res="轻度拥堵"
elif v>10:
    res="中度拥堵"
else:
    res="严重拥堵"

res_prediction=f"<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\\n<已知信息>现在是北京时间{timestamp}，车辆在{location}十字路口未来一小时预测的平均速度是{v:.2f}公里/小时，根据预测的结果，在未来一个小时内，{location}十字路口{res} </已知信息>\\n<问题>请问{location}十字路口拥堵吗?</问题>"

print(res_prediction)

print(f'响应时间：{e-s}')
