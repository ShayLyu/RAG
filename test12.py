import gradio as gr

theme = gr.themes.Ocean(primary_hue="blue", secondary_hue="cyan",neutral_hue="teal")


with gr.Blocks(theme=theme) as chat:  # Ocean theme
    gr.Markdown("## 🌊 Ocean-Themed Chatbot")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type a message...")
    submit_button = gr.Button("Send")
    submit_button.click(lambda x: f"You said: {x}", inputs=user_input, outputs=chatbot)

chat.launch()