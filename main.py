from gradio_web.app import demo

if __name__ == "__main__":
    print("===== 车载座舱轻量化RAG系统启动 =====")
    print("访问地址：http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)