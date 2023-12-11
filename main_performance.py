import argparse
import transformers
from transformers import AutoTokenizer, AutoModel
import platform
import os
import torch
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import requests
from time import time

# 选项
parser = argparse.ArgumentParser(
        description="main performance")
parser.add_argument(
    "--parallel",
    action="store_true",
    help="Whether test model in parallel",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./",
    help="the path to model weights",
)
args = parser.parse_args()

# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu
if args.parallel:
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank==0:
        torch_npu.npu.set_device(0)
    elif local_rank==1:
        torch_npu.npu.set_device(1)
    torch.manual_seed(1)
else:
    DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
    device_id = 0
    if DEVICE_ID is not None:
        device_id = int(DEVICE_ID)
    print(f"user npu:{device_id}")
    torch.npu.set_device(torch.device(f"npu:{device_id}"))

# 使用二进制优化，消除动态shape的编译问题
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
torch.npu.set_option(option)

# 加载模型配置和权重
if args.parallel:
    tokenizer_path = os.path.join(args.model_path, "tokenizer")
    part_model_path = os.path.join(args.model_path, "part_model", str(local_rank))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(part_model_path, trust_remote_code=True).half().npu()
else:
    print("START LOAD!!!!!!!!!")
    tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
    model = AutoModel.from_pretrained("./", trust_remote_code=True).half().npu()
    model.eval()
    print("END LOAD!!!!!!!!!")

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

# 修改transformers的TopKLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    top_k = min(self.top_k, scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    filter_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(indices_to_remove, filter_value)
    # scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopKLogitsWarper.__call__ = __call__

# 修改transformers的TopPLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    cumulative_probs = sorted_logits.softmax(
        dim=-1).cpu().float().cumsum(dim=-1).to(sorted_logits.device).to(sorted_logits.dtype)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    if self.min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    filter_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(indices_to_remove, filter_value)
    # scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopPLogitsWarper.__call__ = __call__

# 优化ND NZ排布，消除transdata
soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [104, 220, 221, 222, 223]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(2)
    print("soc_version:", soc_version, " is 910B, support ND")
else:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(29)
    print("soc_version:", soc_version, " is not 910B, support NZ")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = module.weight.data.npu_format_cast(2)

from pydantic import BaseModel # 引入BaseModel
# 将你的数据模型声明为继承自 BaseModel 的类
class Item(BaseModel):
    prediction:str
    question:str
    timestamp:str
    location:str


def stream(query):
    torch.npu.set_device(torch.device("npu:0"))
    ss = 0
    import time
    start_time = time.time()
    print("api query=:",query)
    model.count = 0
    model.input_generate = 0
    model.model_total = 0
    model.token_total = 0
    model.model_time = 0
    model.token_time = 0
    model.model_first = 0
    model.token_first = 0
    model.pre_processing = 0
    model.post_processing = 0
    #return "success"i
    for response, _  in model.stream_chat(tokenizer, query, history=[]):
        res = response[(ss):]
        #print("rs: ", res)
        for j in res:
            yield j
        ss = len(response)
    end = time.time()
    print("predict time:",end-start_time)
async def stream_chat(item:Item):
    if item.prediction=="1":
        print("现在开始预测")
        url='http://localhost:1125/TrafficFlowForcast'
        data={'timestamp':item.timestamp,'location':item.location}
        #s=time()
        r=requests.post(json=data,url=url).json()
        #e=time()
        print(r)
        #print(f'响应时间:{e-s}')
        split_list =   r['result'].split(";")
        sum=0.0
        for sl in split_list:
            print(sl)
            sum =sum+float(sl)
        sum=sum/12
        
        print(sum)

        
        
        #def average_speed(L, N):
            # 使用车辆的实际平均速度的公式计算速度
            #v = L / N *12
            #return v
        
        #L = 500 # 路段的长度，单位为米
        #N = sum # 路段上的车辆数，单位为辆
        
        #v = average_speed(L, N)
        #print(f"车辆的实际平均速度是{v:.2f}公里/小时")

        #if v>40:
            #resd="通畅"
        #elif v>30:
            #resd="比较通畅"
        #elif v>20:
            #resd="轻度拥堵"
        #elif v>10:
            #resd="中度拥堵"
        #else:
            #resd="严重拥堵"
        if sum>254:
            resd="严重拥堵"
        elif sum>208:
            resd="中度拥堵"
        elif sum>160:
            resd="轻度拥堵"
        else:
            resd="通畅"


        #print(resd)
        #res_prediction=f"<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\\n<已知信息>现在是北京时间{item.timestamp}，车辆在{item.location}十字路口未来一小时预测的平均速度是{v:.2f}公里/小时，根据预测的结果，在未来一个小时内，{item.location}十字路口{resd} </已知信息>\\n<问题>请问{item.location}十字路口拥堵吗?</问题>"
        res_prediction=f"现在是北京时间{item.timestamp}，车辆在{item.location}十字路口未来一小时预测的车流量是{sum:.0f}辆/小时，根据预测的结果，在未来一个小时内，{item.location}十字路口{resd}"
        ress=iter(res_prediction)
        return StreamingResponse(ress, media_type="text/plain")
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

