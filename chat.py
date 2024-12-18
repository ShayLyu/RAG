import os
from openai import OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.vectorstores import FAISS
from create_kb import create_tmp_kb

DB_PATH = "VectorStore"
TMP_NAME = "tmp_abcd"

EMBED_MODEL = DashScopeEmbeddings(
    model="text-embedding-v2"
)

# 定义全局变量
model = "qwen-max"
temperature = 0.85
max_tokens = 1024
history_round = 3
db_name = "some_default_db"
similarity_threshold = 0.2
chunk_cnt = 5

def update_global_params(new_model, new_temperature, new_max_tokens, new_history_round, new_db_name, new_similarity_threshold, new_chunk_cnt):
    global model, temperature, max_tokens, history_round, db_name, similarity_threshold, chunk_cnt
    model = new_model
    temperature = new_temperature
    max_tokens = new_max_tokens
    history_round = new_history_round
    db_name = new_db_name
    similarity_threshold = new_similarity_threshold
    chunk_cnt = new_chunk_cnt

def get_model_response(
        multi_modal_input,
        history,
        model,
        temperature,
        max_tokens,
        history_round,
        db_name,
        similarity_threshold,
        chunk_cnt
):
    # 在函数内部调用update_global_params将传入的参数写回全局变量
    update_global_params(model, temperature, max_tokens,history_round, db_name, similarity_threshold, chunk_cnt)

    # prompt为用户最新的一条对话
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']

    # 如果tmp文件目录存在，则使用TMP_NAME数据库；否则如果用户上传了文件，也创建临时数据库
    if os.path.exists(os.path.join("File", TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME

    print(f"prompt: {prompt}, tmp_files: {tmp_files}, db_name: {db_name}")

    # 尝试加载FAISS向量数据库
    try:
        vectorstore_path = os.path.join(DB_PATH, db_name)
        if not os.path.exists(vectorstore_path):
            # 若数据库不存在，直接使用prompt本身进行回复
            raise Exception("Database does not exist.")

        # 加载FAISS向量数据库
        vectorstore = FAISS.load_local(vectorstore_path, EMBED_MODEL, allow_dangerous_deserialization=True)

        # 使用similarity_search_with_score获取文档和相似度分数
        docs_and_scores = vectorstore.similarity_search_with_score(prompt, k=20)
        print(f"原始检索结果: {docs_and_scores}")

        # 本示例中不再使用rerank逻辑，直接使用检索结果。
        # docs_and_scores为[(Document, score), ...]
        # 根据相似度分数和chunk_cnt、similarity_threshold进行过滤和截断
        filtered_results = [
                               (doc, score) for doc, score in docs_and_scores
                               if score >= similarity_threshold
                           ][:chunk_cnt]

        chunk_text = ""
        chunk_show = ""
        for i, (doc, score) in enumerate(filtered_results):
            chunk_text += f"## {i + 1}:\n {doc.page_content}\n"
            chunk_show += f"## {i + 1}:\n {doc.page_content}\nscore: {round(score, 2)}\n"

        if chunk_text.strip():
            prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
        else:
            # 如果没有符合阈值的chunk，则直接使用用户的prompt
            prompt_template = prompt
            chunk_show = ""
    except Exception as e:
        print(f"异常信息：{e}")
        # 出现异常则直接使用原始prompt
        prompt_template = prompt
        chunk_show = ""

    # 重置当前回答为空
    history[-1][-1] = ""

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}

    # 构造对话上下文
    messages = []
    history_round = min(len(history), history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round + i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round + i][1]})
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    assistant_response = ""
    for chunk in completion:
        assistant_response += chunk.choices[0].delta.content
        history[-1][-1] = assistant_response
        yield history, chunk_show

def get_model_response_for_user(history):
    # history格式为 [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]

    # 从 history 中获取用户最新消息
    # 用户最新消息为 history 中最后一个role为"user"的条目
    # 由于我们每次都append用户消息，然后接模型回答，因此最后一个消息一定是用户的
    prompt = history[-1]["content"]
    print(f"prompt: {prompt}, db_name: {db_name}")

    # 以下是向量数据库检索与模板处理的逻辑略
    try:
        vectorstore_path = os.path.join(DB_PATH, db_name)
        if not os.path.exists(vectorstore_path):
            raise Exception("Database does not exist.")

        # 这里是伪代码，仅供示意，实际请根据您的RAG逻辑修改
        # vectorstore = FAISS.load_local(vectorstore_path, EMBED_MODEL, allow_dangerous_deserialization=True)
        # docs_and_scores = vectorstore.similarity_search_with_score(prompt, k=20)

        # 简化示例，只返回prompt本身作为回答
        prompt_template = prompt

    except Exception as e:
        print(f"异常信息：{e}")
        prompt_template = prompt

    # 调用OpenAI模型生成回答
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    used_history_round = min(len(history), history_round)

    # 构造messages
    messages = [system_message]
    # 将最近N轮对话加入上下文
    # history已是[{role:"user"|"assistant", content:"..."}]格式
    # 根据used_history_round截取最后N条消息
    relevant_messages = history[-used_history_round*2:] if used_history_round*2 <= len(history) else history
    messages.extend(relevant_messages)
    messages.append({'role': 'user', 'content': prompt_template})
    print(messages)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    assistant_response = ""
    for chunk in completion:
        assistant_response += chunk.choices[0].delta.content
        # 更新history中assistant回答的条目
        # 当回答刚开始生成时还没有assistant消息条目，需要append新的assistant消息
        # 为了流式更新，每次chunk更新assistant消息的最后一条
        if len(history) > 0 and history[-1]["role"] == "assistant":
            # 已有assistant消息条目，更新其content
            history[-1]["content"] = assistant_response
        else:
            # 没有assistant消息条目，创建一个新的assistant消息条目
            history.append({"role": "assistant", "content": assistant_response})
        yield history

def response_for_users(message, history):
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