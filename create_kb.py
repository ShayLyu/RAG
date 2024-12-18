#####################################
######       创建知识库         #######
#####################################
import gradio as gr
import os
import shutil
from typing import List
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS

# 文件路径设定
DB_PATH = "VectorStore"
STRUCTURED_FILE_PATH = "File/Structured"
UNSTRUCTURED_FILE_PATH = "File/Unstructured"
TMP_NAME = "tmp_abcd"

# 使用 DashScopeEmbeddings 示例
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2"
)


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """从指定文件夹中加载文本文件，并创建Document列表。"""
    documents = []
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            full_path = os.path.join(directory_path, file_name)
            if os.path.isfile(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # 创建Document对象，可根据需要在metadata中添加文件名等信息
                documents.append(Document(page_content=text, metadata={"source": file_name}))
    return documents


# 刷新知识库列表
def refresh_knowledge_base():
    return os.listdir(DB_PATH)


# 创建非结构化向量数据库（使用FAISS）
def create_unstructured_db(db_name: str, label_name: List[str]):
    print(f"知识库名称为：{db_name}，类目名称为：{label_name}")
    if not label_name:
        gr.Info("没有选择类目")
        return
    if len(db_name.strip()) == 0:
        gr.Info("没有命名知识库")
        return
    # 判断是否存在同名向量数据库
    if db_name in os.listdir(DB_PATH):
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
        return
    else:
        gr.Info("正在创建知识库，请等待知识库创建成功信息显示后前往RAG问答")
        documents = []
        for label in label_name:
            label_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
            documents.extend(load_documents_from_directory(label_path))

        # 创建向量数据库（FAISS 不支持 persist_directory，需要使用 save_local）
        db_path = os.path.join(DB_PATH, db_name)
        if not os.path.exists(db_path):
            os.mkdir(db_path)
        vectorstore = FAISS.from_documents(documents, embeddings)
        # 保存向量索引到本地
        vectorstore.save_local(db_path)

        gr.Info("知识库创建成功，可前往RAG问答进行提问")


# 创建结构化向量数据库（使用FAISS）
def create_structured_db(db_name: str, data_table: List[str]):
    print(f"知识库名称为：{db_name}，数据表名称为：{data_table}")
    if not data_table:
        gr.Info("没有选择数据表")
        return
    if len(db_name.strip()) == 0:
        gr.Info("没有命名知识库")
        return
    # 判断是否存在同名向量数据库
    if db_name in os.listdir(DB_PATH):
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
        return
    else:
        gr.Info("正在创建知识库，请等待知识库创建成功信息显示后前往RAG问答")
        documents = []
        for table in data_table:
            label_path = os.path.join(STRUCTURED_FILE_PATH, table)
            # 将文件中每一行作为一个document
            if os.path.exists(label_path):
                for file_name in os.listdir(label_path):
                    full_path = os.path.join(label_path, file_name)
                    if os.path.isfile(full_path):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                documents.append(Document(page_content=line, metadata={"source": file_name}))

        db_path = os.path.join(DB_PATH, db_name)
        if not os.path.exists(db_path):
            os.mkdir(db_path)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(db_path)

        gr.Info("知识库创建成功，可前往RAG问答进行提问")


# 删除指定名称知识库
def delete_db(db_name: str):
    if db_name:
        folder_path = os.path.join(DB_PATH, db_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            gr.Info(f"已成功删除{db_name}知识库")
            print(f"已成功删除{db_name}知识库")
        else:
            gr.Info(f"{db_name}知识库不存在")
            print(f"{db_name}知识库不存在")


# 实时更新知识库列表
def update_knowledge_base():
    return gr.update(choices=os.listdir(DB_PATH))


# 临时文件创建知识库（使用FAISS）
def create_tmp_kb(files: List[str]):
    tmp_dir = os.path.join("File", TMP_NAME)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, os.path.join(tmp_dir, file_name))

    documents = load_documents_from_directory(tmp_dir)
    db_path = os.path.join(DB_PATH, TMP_NAME)
    if not os.path.exists(db_path):
        os.mkdir(db_path)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(db_path)


# 清除tmp文件夹内容
def clear_tmp():
    tmp_file_path = os.path.join("File", TMP_NAME)
    tmp_db_path = os.path.join(DB_PATH, TMP_NAME)

    if os.path.exists(tmp_file_path):
        shutil.rmtree(tmp_file_path)
    if os.path.exists(tmp_db_path):
        shutil.rmtree(tmp_db_path)
