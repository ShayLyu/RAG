import os
from openai import OpenAI
import gradio as gr

# Define global variables
model = "qwen-max"
temperature = 0.85
max_tokens = 1024


def response_for_user(message, history):
    """
    Generates a response for the user based on message and chat history.
    Args:
        message (str): User's input message.
        history (list): Chat history in Gradio format.
    Yields:
        str: Streaming response content.
    """
    try:
        # Print input message and history for debugging
        print(f"Message: {message}")
        print(f"History: {history}")

        # Validate message
        if not message.strip():
            yield "Your input is empty. Please provide a valid message."
            return

        # Prepare chat history for OpenAI API
        chat_history = [
            {"role": entry["role"], "content": entry["content"]}
            for entry in history if "content" in entry and entry["content"]
        ]
        chat_history.append({"role": "user", "content": message})
        print(f"chat_history: {chat_history}")

        # Ensure API key is available
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            yield "Error: API key not found. Please set the DASHSCOPE_API_KEY environment variable."
            return

        # Configure OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # Call OpenAI API with streaming
        completion = client.chat.completions.create(
            model=model,
            messages=chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        # Accumulate response content
        full_response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                # Update the history incrementally
                yield {"role": "assistant", "content": full_response}
    except Exception as e:
        print(f"Error: {e}")
        yield f"An error occurred: {e}"


# Launch the Gradio chat interface
gr.ChatInterface(
    fn=response_for_user,  # Generator function
    type="messages",       # Message-based interaction
    title="AI Chatbot Interface", # Set interface title
    description="Ask any question and receive AI-generated responses in real-time."
).launch()
