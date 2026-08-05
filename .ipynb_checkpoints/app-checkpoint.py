import gradio as gr
from speech_asr.whisper_asr import audio2text
from intent_cls.infer_intent import get_intent
from retriever.rerank import hybrid_retrieve
from llm_infer.rag_llm import generate_rag_answer

def pipeline(audio, text_input):
    # 1.语音转文本
    user_text = ""
    if audio is not None:
        sr, audio_arr = audio
        user_text = audio2text(audio_arr, sr)
    if text_input.strip() != "":
        user_text = text_input.strip()
    if not user_text:
        return "", "", "无输入", "", ""

    # 2.意图识别分流
    intent = get_intent(user_text)
    if intent == "control":
        return user_text, intent, "识别为座舱控制指令，无需检索", "", "已下发座舱控制指令"
    if intent == "nav":
        return user_text, intent, "导航指令，跳转导航模块", "", "请说出目的地"
    if intent == "chat_none":
        return user_text, intent, "无检索", "", "我仅能解答车辆座舱相关问题"

    # 3.知识问答：执行RAG
    ctx_str, ctx_list = hybrid_retrieve(user_text)
    # 4.LLM生成
    ans = generate_rag_answer(user_text, ctx_str)
    return user_text, intent, ctx_str, ctx_list, ans

with gr.Blocks(title="车载座舱轻量化RAG调试后台") as demo:
    gr.Markdown("# 车载座舱RAG问答系统 可视化调试面板")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(label="麦克风语音输入", type="numpy")
            text_in = gr.Textbox(label="手动输入指令", placeholder="如何打开座椅加热？")
            submit_btn = gr.Button("执行问答")
        with gr.Column(scale=2):
            out_text = gr.Textbox(label="ASR识别文本")
            out_intent = gr.Textbox(label="识别意图")
            out_ctx = gr.Textbox(label="检索参考资料", lines=8)
            out_ans = gr.Textbox(label="座舱助手回答", lines=4)
    submit_btn.click(
        fn=pipeline,
        inputs=[audio_in, text_in],
        outputs=[out_text, out_intent, out_ctx, gr.State(), out_ans]
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)