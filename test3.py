import random

import gradio as gr
from gradio.themes import Color, Base


# 定义响应函数
def random_response(message, history):
    return random.choice(["Yes", "No"])
#灰色。2E3148
gray = Color(
    name="gray",#
    c50="#f9fafb",#白。background_fill_secondary table_odd_background_fill
    c100="#2E3148", # 1
    c200="#e5e7eb",#button_secondary_background_fill border_color_primary block_label_text_color_dark block_title_text_color_dark
    c300="#d1d5db",
    c400="#9ca3af",
    c500="#6b7280",
    c600="#4b5563",
    c700="#374151",
    c800="#1f2937",
    c900="#111827",
    c950="#0b0f19",
)

examples = [
    {"text": "石墨烯是什么?"},
    {"text": "石墨烯如何制备？"},
    {"text": "石墨烯有什么应用?"},
]
# with gr.Blocks(css=".gradio-container {background-color: red}") as demo:

# 定义自定义主题
import gradio as gr
from gradio.themes import Base


# 创建自定义主题实例
unified_dark_theme = UnifiedDarkTheme()


# 创建 ChatInterface
chat = gr.ChatInterface(
    fn=random_response,
    # type="messages",
    title="石墨烯智能问答系统",
    examples=examples,
    description="输入您的问题，系统将提供智能回答。",
    fill_width=True,
    fill_height=True,

    chatbot=gr.Chatbot(
                       placeholder="<strong>您也许想问:</strong>",
                       avatar_images=("./image/user-tx.jpg","./image/kf-tx.jpg"),
                       type="messages"),
    css=".gradio-container {background-color: #150F37}",
    theme=UnifiedDarkTheme()

)

# 启动界面并注入自定义 CSS
if __name__ == "__main__":
    chat.launch()
