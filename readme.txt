五、完整落地执行步骤（按顺序跑）
把车辆使用手册 PDF 放入 data/raw/
解析文档：python data_process/parse_pdf.py
文本分块：python data_process/build_chunk.py
构建 FAISS 向量库：python vector_store/build_faiss.py
（可选）训练意图分类：python intent_cls/train_intent.py
（可选）LoRA 领域微调：python llm_infer/train_lora.py
启动 Gradio 可视化系统：python main.py
浏览器打开 http://127.0.0.1:7860 即可语音 / 文本测试完整 RAG 链路


启动训练：python intent_cls/train_intent.py
训练完成后自动生成 intent_fineturn 权重文件夹，infer_intent.py 可直接加载推理

车载量产 FastAPI 推理服务（无 Gradio、车机离线部署专用）

FastAPI 使用说明
安装依赖：pip install fastapi uvicorn soundfile
启动推理服务：python car_api.py
接口访问地址：
健康检测：http://车机IP:7890/health
文本问答 POST 接口：http://IP:7890/chat_text，传参 query="如何打开座椅加热"
音频上传接口：http://IP:7890/chat_audio，form-data 上传 wav 音频
自动接口文档：http://IP:7890/docs，可在线调试所有接口

check_qa.py 检查QA对也就是qa_train/train.jsonl 格式是否正确

demo 3~8pdf   300-1000 chunk  适配 faiss  cpu跑无压力
POC演示   10~20 pdf    2000-5000chunk
量产车机离线部署   30~80 pdf    1万chunk 以内
为什么不能无限堆 PDF？

1. 车机硬件算力 & 内存（最关键约束）
你的系统是端侧车载轻量化 RAG，不是云端大向量库：
chunk 越多，向量库体积越大、加载内存占用越高；
超过 1 万条文本块，低端车机开机加载向量库耗时 > 2s，问答延迟突破 1s，不符合语音实时交互要求；
4bit 量化 Qwen1.8B+FAISS，最优 chunk 规模：3000～8000 条。
2. 问答场景覆盖需求
只做座舱语音问答（空调、座椅、车机、故障），不需要整车底盘、发动机深度维修资料；
只聚焦座舱相关内容：少量手册足够；
若要覆盖整车故障、维修、保养，需补充更多维修手册 PDF。
3. 检索准确率（文档过多反而效果变差）
PDF 数量过多会带来大量冗余、相似文本块：
检索时无关片段增多，重排模型过滤压力变大；
大量无关上下文送入 LLM，增加幻觉概率；
高质量 10 份座舱专项手册 > 100 份杂乱整车手册。
4. 迭代与维护成本
车型 OTA 更新、新款上市，需要新增 PDF、重新分块、重建向量库；
文档越少，更新、清洗、重构建向量库速度越快，车机后台增量更新负担更低


数据从这个网站获取
分类齐全：整车用户手册、座舱多媒体手册、故障码速查手册；
1. CarOBook 车主随身手册（国内最全中文车型库）
网址：https://www.carobook.com

QA对 训练lora数据来源，使用来自raw切分后的chunk块

下面是一段汽车用户手册文本，请基于这段内容生成5组车主真实口语化问答对。
要求：
1. question是车主日常语音提问，简短自然；
2. answer严格摘抄原文，不扩写、不编造，简短1-2句话；
3. 输出纯JSON数组，格式[{"question":"xxx","answer":"xxx"},...]，不要多余文字。
文本：
{此处粘贴chunk文本}


人工制作QA对
座舱控制类（空调、座椅、车窗、灯光、音量）
语音车机类（蓝牙、导航、语音助手、投屏）
故障码 / 报警类（qa_know 高频问题）

demo： 50-200条
POC 500-1500
量产 3000-8000

QA数据对 数据清洗 & 过滤规则（关键，避免训练崩坏）
1.剔除过长问答：问答合计超过 500 字符直接删除；
2.删除重复 question：相同提问只保留 1 条；
3.淘汰模糊回答：answer 不能出现 “可能、大概、建议检查” 等模糊话术，必须精准；
4.剔除无关内容：底盘、发动机、维修大修内容，只保留座舱、车机、故障报警；
5.禁止英文、特殊符号、换行过多，文本干净简洁。


目录和前面 demo 保持兼容，删掉 gradio_web 文件夹，
新增车机部署脚本、守护进程配置、配置文件、启动 / 停止脚

车机上不执行训练脚本；训练、PDF 解析、建向量库全部在开发 PC 完成，
把产出：faiss_index、intent_lora、lora_car 拷贝到车机对应目录。


开发机→车机打包部署流程【非常关键】
所有训练、PDF 解析、构建向量库全部在 PC 开发机完成，车机只运行推理服务。
1. 开发机执行流程
bash
#1 放入原厂PDF
#2解析手册
python data_process/parse_pdf.py
#3文本分块
python data_process/build_chunk.py
#4构建faiss向量库
python vector_store/build_faiss.py
#5训练意图分类
python intent_cls/train_intent.py
#6准备qa_train/train.jsonl，校验
python check_qa.py
#7LoRA微调
python llm_infer/train_lora.py

2. 拷贝到车机的文件 / 文件夹清单
plaintext
data/chunks/chunks.jsonl
vector_store/faiss_index/
intent_cls/intent_lora/
llm_infer/lora_car/
所有py源码
requirements_offline.txt
start_service.sh stop_service.sh

3. 车机上操作
bash
pip3 install -r requirements_offline.txt
chmod +x start_service.sh stop_service.sh
./start_service.sh


测试接口：访问 http://车机IP:7890/docs

接口地址：
健康检测：http://127.0.0.1:7890/health
接口文档：http://127.0.0.1:7890/docs
📡 API 接口
POST /chat_text：文本问答
POST /chat_audio：上传 Wav 音频，语音问答
GET /health：服务状态自检