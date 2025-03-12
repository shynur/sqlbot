import numpy
import torch
import transformers
import faiss
import numpy

# 检测 CUDA 是否可用, 并设置设备
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("sql_doc.txt", "r", encoding="utf-8") as f:
    sql_doc: str = f.read()
# 按空行切分文档:
sql_doc_chunks: list[str] = [
    chunk.strip() for chunk in sql_doc.split("\n\n") if chunk.strip()
]

# 初始化 transformers 模型和 tokenizer:
rag_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
rag_tokenizer = transformers.AutoTokenizer.from_pretrained(rag_model_name)
rag_model = transformers.AutoModel.from_pretrained(rag_model_name)
# 移动模型到 device:
rag_model.to()
rag_model.eval()


def compute_embedding(text: str) -> numpy.ndarray:
    """利用 transformers 模型对输入文本计算 embedding，
    使用简单的平均池化策略（考虑 attention mask）。
    """
    encoded_input: dict[str, torch.Tensor] = rag_tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # 将所有 tensor 移动到 device 上
    encoded_input = {k: v.to(torch_device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = rag_model(**encoded_input)
    # 获取 token 层输出，进行平均池化，注意考虑 attention mask
    token_embeddings: torch.Tensor = (
        model_output.last_hidden_state
    )  # [batch_size, seq_len, hidden_size]
    attention_mask: torch.Tensor = encoded_input["attention_mask"]
    input_mask_expanded: torch.Tensor = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    embedding: torch.Tensor = torch.sum(
        token_embeddings * input_mask_expanded, dim=1
    ) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # 转为 numpy 数组并归一化，先将 tensor 移动到 CPU
    embedding: numpy.ndarray = embedding.cpu().numpy()
    norm: numpy.ndarray = numpy.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / norm


# 3. 计算每个文档片段的 embedding，并建立 FAISS 索引
chunk_embeddings: list[numpy.ndarray] = [
    compute_embedding(chunk) for chunk in sql_doc_chunks
]
# 每个 embedding 的 shape 为 (1, dim)，堆叠成 (n_chunks, dim)
chunk_embeddings: numpy.ndarray = numpy.vstack(chunk_embeddings)
dim: int = chunk_embeddings.shape[1]

# 使用 FAISS 建立 L2 距离索引
index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
index.add(chunk_embeddings)


def retrieve_context(query: str, top_k) -> list[str]:
    """
    给定用户查询，计算其 embedding 并利用 FAISS 检索 top_k 个最相关的文档片段。
    """
    query_embedding: numpy.ndarray = compute_embedding(query)  # shape: (1, dim)
    distances: numpy.ndarray
    indices: numpy.ndarray
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks: list[str] = [sql_doc_chunks[idx] for idx in indices[0]]
    return retrieved_chunks


# 4. 根据用户查询生成最终 prompt（仅生成 prompt，不调用 openai API）
user_query: str = "请帮我生成一个查询所有用户订单总金额大于1000的 SQL 语句"
retrieved_chunks: list[str] = retrieve_context(user_query, top_k=30)

# 构造 prompt，将检索到的文档片段与用户查询整合
prompt: str = "你是一个 SQL 语法专家。以下是部分 SQL 语法文档内容：\n"
for chunk in retrieved_chunks:
    prompt += f"\n---\n{chunk}\n"
prompt += f"\n根据以上文档内容和用户需求，请生成对应的 SQL 语句。\n用户需求: {user_query}\nSQL语句:"

print("生成的 Prompt:\n")
print(prompt)
