import argparse
import transformers
from transformers import AutoTokenizer, AutoModel
import os
import torch
import io

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import requests

from pydantic import BaseModel # 引入BaseModel
# 将你的数据模型声明为继承自 BaseModel 的类
class Item(BaseModel):
    prediction:str
    question:str
    timestamp:str
    location:str

ss = 0
#tokenizer = AutoTokenizer.from_pretrained("/data/pengwang/TransGPT/mer_sft/chatglm2/batchsize_16_epoch_20_lorarank_32_loraalpha_64", trust_remote_code=True)
#model = AutoModel.from_pretrained("/data/pengwang/TransGPT/mer_sft/chatglm2/batchsize_16_epoch_20_lorarank_32_loraalpha_64", trust_remote_code=True, device='cuda')

def stream(query):
    #for response, _  in model.stream_chat(tokenizer, query, history=[]):
        #res = response[(ss):]
        #for j in res:
            #yield j
        #ss= len(response)
    res = query
    for j in res:
        yield j
    #ss= len(response)

    #s = b"success"
    #return io.BytesIO(s)
async def stream_chat(item:Item):    
    if item.prediction=="1":
        print("现在开始预测")
        url='http://localhost:1125/TrafficFlowForcast'
        data={'timestamp':item.timestamp,'location':item.location}
        #s=time()
        r=requests.post(json=data,url=url).json()
        #e=time()
        #print(r['result'])
        #print(f'响应时间:{e-s}')
        split_list =   r['result'].split(";")
        sum=0.0
        for sl in split_list:
            print(sl)
            sum =sum+float(sl)
        sum=sum/12
        print(sum)
        
        
        def average_speed(L, N):
            # 使用车辆的实际平均速度的公式计算速度
            v = L / N *12
            return v
        
        L = 500 # 路段的长度，单位为米
        N = sum # 路段上的车辆数，单位为辆
        
        v = average_speed(L, N)
        #print(f"车辆的实际平均速度是{v:.2f}公里/小时")
        
        if v>40:
            resd="通畅"
        elif v>30:
            resd="比较通畅"
        elif v>20:
            resd="轻度拥堵"
        elif v>10:
            resd="中度拥堵"
        else:
            resd="严重拥堵"
        
        #print(resd)
        res_prediction=f"<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\\n<已知信息>现在是北京时间{item.timestamp}，车辆在{item.location}十字路口未来一小时预测的平均速度是{v:.2f}公里/小时，根据预测的结果，在未来一个小时内，{item.location}十字路口{resd} </已知信息>\\n<问题>请问{item.location}十字路口拥堵吗?</问题>"
        return StreamingResponse(stream(res_prediction), media_type="text/plain")
    print("现在是问答")
    return StreamingResponse(stream(item.question), media_type="text/plain")
def main():        
    app = FastAPI()
    # 允许跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )    
    app.post('/model_api')(stream_chat)
    uvicorn.run(app, host='0.0.0.0', port=7860)    

if __name__ == "__main__":    
    main()
