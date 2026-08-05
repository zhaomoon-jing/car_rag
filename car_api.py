from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import soundfile as sf
import tempfile

# 导入全链路模块
from speech_asr.whisper_asr import audio2text
from intent_cls.infer_intent import get_intent
from retriever.rerank import hybrid_retrieve
from llm_infer.rag_llm import generate_rag_answer

# 初始化服务
app = FastAPI(title="车载座舱RAG离线推理API")

# 跨域配置，适配车机中控UI请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 响应数据格式模板
class RAGResponse:
    def __init__(self):
        self.user_text: str = ""          # ASR识别文本
        self.intent: str = ""             # 意图分类结果
        self.intent_desc: str = ""        # 意图说明
        self.reference_context: str = ""  # 检索参考资料
        self.answer: str = ""             # 助手回答

# 接口1：纯文本指令问答（车机文字搜索/语音转文字后调用）
@app.post("/chat_text")
async def chat_text(query: str):
    resp = RAGResponse()
    resp.user_text = query.strip()
    if not resp.user_text:
        return {"code": 400, "msg": "输入不能为空", "data": resp.__dict__}

    # 意图识别分流
    resp.intent = get_intent(resp.user_text)
    if resp.intent == "control":
        resp.intent_desc = "座舱硬件控制指令，直接下发控制信号，无需知识库检索"
        resp.answer = "已收到控制指令，正在执行座舱硬件操作"
        return {"code": 200, "msg": "success", "data": resp.__dict__}
    elif resp.intent == "nav":
        resp.intent_desc = "导航类指令，跳转导航模块处理"
        resp.answer = "请告知需要导航的目的地"
        return {"code": 200, "msg": "success", "data": resp.__dict__}
    elif resp.intent == "chat_none":
        resp.intent_desc = "无关闲聊，仅支持车辆座舱相关问答"
        resp.answer = "我是车载座舱助手，仅能解答车辆功能、故障、使用手册相关问题"
        return {"code": 200, "msg": "success", "data": resp.__dict__}

    # 知识问答：执行RAG检索+大模型生成
    resp.reference_context, _ = hybrid_retrieve(resp.user_text)
    resp.answer = generate_rag_answer(resp.user_text, resp.reference_context)
    resp.intent_desc = "车辆知识库问答，已检索手册资料生成回答"
    return {"code": 200, "msg": "success", "data": resp.__dict__}

# 接口2：上传音频文件，自动ASR转文字后问答（车载麦克风录音上传）
@app.post("/chat_audio")
async def chat_audio(audio_file: UploadFile = File(...)):
    resp = RAGResponse()
    # 临时保存音频文件并读取
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio_file.read())
        tmp_path = tmp.name
    data, sr = sf.read(tmp_path)
    resp.user_text = audio2text(data, sr)

    # 复用文本问答逻辑
    resp.intent = get_intent(resp.user_text)
    if resp.intent == "control":
        resp.intent_desc = "座舱硬件控制指令，直接下发控制信号，无需知识库检索"
        resp.answer = "已收到控制指令，正在执行座舱硬件操作"
    elif resp.intent == "nav":
        resp.intent_desc = "导航类指令，跳转导航模块处理"
        resp.answer = "请告知需要导航的目的地"
    elif resp.intent == "chat_none":
        resp.intent_desc = "无关闲聊，仅支持车辆座舱相关问答"
        resp.answer = "我是车载座舱助手，仅能解答车辆功能、故障、使用手册相关问题"
    else:
        resp.reference_context, _ = hybrid_retrieve(resp.user_text)
        resp.answer = generate_rag_answer(resp.user_text, resp.reference_context)
        resp.intent_desc = "车辆知识库问答，已检索手册资料生成回答"

    return {"code": 200, "msg": "success", "data": resp.__dict__}

# 健康检查接口，车机开机自检调用
@app.get("/health")
async def health():
    return {"status": "running", "desc": "车载RAG推理服务正常运行"}

if __name__ == "__main__":
    # 车机局域网开放，端口7890
    uvicorn.run("car_api:app", host="0.0.0.0", port=7890, log_level="info")