#! /bin/python3.13
# -*- coding: utf-8 -*-

import re
import sqlite3
import logging
import json
import concurrent.futures
import threading
import os
import contextlib
import time
from typing import (
    Any,
    Callable,
    Iterable,
)

import openai
import numpy
import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import sqlparse
import plotly.graph_objects
import plotly.subplots
import torch
import transformers
import faiss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

db = sqlite3.connect(
    "eg.sqlite3",
    autocommit=False,
    # 我们会在多线程中访问数据库, 因此需要关闭检查线程是否相同的功能.
    check_same_thread=False,
)
db_lock = threading.Lock()  # 简单起见, 无论读写都默认加锁.
db.row_factory = lambda cursor, row: {
    field: value
    for field, value in zip(
        [field_info[0] for field_info in cursor.description],
        row,
    )
}


class Prompt:
    def __init__(self, query: str = ""):
        self.query: str = query
        self.background: list[str] = []
        self.knowledge: list[str] = []

    def __str__(self) -> str:
        """返回一个可以直接提供给 LLM 的 prompt."""

        self.knowledge = [
            knowledge.strip() for knowledge in self.knowledge if knowledge.strip()
        ]
        self.background = [
            background.strip() for background in self.background if background.strip()
        ]

        knowledge: str = (
            (self.knowledge or "")
            and f"""
以下是一些前置知识:

{
    '\n\n'.join(
        '\n'.join(
            f'> {line}' for line in knowledge.splitlines()
        ) for knowledge in self.knowledge
    )
}

_______________________________________________________________________________
        """.strip()
        )

        background: str = (
            (self.background or "")
            and f"""
已知的背景信息:

{
    '\n\n'.join(
        '\n'.join(
            f'> {line}' for line in background.splitlines()
        ) for background in self.background
    )
}
        """.strip()
        )

        return f"""
{knowledge}

{background}

{self.query}
        """.strip()


class LLM:
    def __init__(
        self,
        *,
        model_name: str,
        API_KEY: str,
        base_url: str,
    ):
        self.model_name: str = model_name
        self.client = openai.OpenAI(
            api_key=os.getenv(API_KEY),
            base_url=base_url,
        )

    def get_response(
        self,
        *,
        messages: Iterable[dict[str, Any]],
        response_format=openai.NOT_GIVEN,
    ) -> str:
        """获取 AI 的回答.

        不包含任何空白符.
        """

        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
            )
            .choices[0]
            .message.content.strip()
        )

    @staticmethod
    def get_ai(
        role: str = "chatter",
        *,
        _locals: dict[str, "LLM"] = {},
    ) -> "LLM":
        """根据 ROLE 获取不同特长的 AI client.

        我们将不同的 model 缓存到 _LOCALS 中, 以便多次使用.
        """

        if "qwen_chatter" not in _locals:
            _locals["qwen_chatter"] = LLM(
                model_name="qwen2.5-1.5b-instruct",
                API_KEY="TongYiQianWen_API_key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        if "qwen_coder" not in _locals:
            _locals["qwen_coder"] = LLM(
                model_name="qwen2.5-coder-1.5b-instruct",
                API_KEY="TongYiQianWen_API_key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

        match role:
            case "chatter":
                return _locals["qwen_chatter"]
            case "coder":
                return _locals["qwen_coder"]
            case _:
                raise ValueError(f"Invalid role: {role}")


class RAG:
    # 检测 CUDA 是否可用, 并设置设备:
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 transformers 模型和 tokenizer:
    rag_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_tokenizer = transformers.AutoTokenizer.from_pretrained(rag_model_name)
    rag_model = transformers.AutoModel.from_pretrained(rag_model_name)

    rag_model.to()  # 移动模型到 device.
    rag_model.eval()

    def __init__(
        self,
        doc_path: str,
        sep: str = "\n\n",
    ):
        """加载文档并按指定分隔符切分文档."""

        with open(doc_path, "r", encoding="utf-8") as f:
            doc: str = f.read()

        # 按空行切分文档:
        self.doc_chunks: list[str] = [
            chunk.strip() for chunk in doc.split(sep) if chunk.strip()
        ]

        # 计算每个文档片段的 embedding, 并建立 FAISS 索引:
        chunk_embeddings_list: list[numpy.ndarray] = [
            self.__class__.compute_embedding(chunk) for chunk in self.doc_chunks
        ]

        # 每个 embedding 的 shape 为 (1, dim), 堆叠成 (n_chunks, dim):
        chunk_embeddings: numpy.ndarray = numpy.vstack(chunk_embeddings_list)
        dim: int = chunk_embeddings.shape[1]

        # 使用 FAISS 建立 L2 距离索引:
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
        self.index.add(chunk_embeddings)

    @classmethod
    def compute_embedding(cls, text: str) -> numpy.ndarray:
        """利用 transformers 模型对输入文本计算 embedding.

        使用简单的平均池化策略 (考虑 attention mask).
        """

        encoded_input = cls.rag_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # 将所有 tensor 移动到 device 上:
        encoded_input = {k: v.to(cls.torch_device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = cls.rag_model(**encoded_input)

        # 获取 token 层输出, 进行平均池化 (注意考虑 attention mask).
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

        # 转为 numpy 数组并归一化，先将 tensor 移动到 CPU:
        embedding_np: numpy.ndarray = embedding.cpu().numpy()
        norm: numpy.ndarray = numpy.linalg.norm(embedding_np, axis=1, keepdims=True)
        return embedding_np / norm

    def retrieve_context(
        self,
        query: str,
        top_k: int,
    ) -> list[str]:
        """给定用户查询, 检索 TOP_K 个最相关的文档片段."""

        query_embedding: numpy.ndarray = self.__class__.compute_embedding(
            query
        )  # shape: (1, dim)

        distances: numpy.ndarray
        indices: numpy.ndarray
        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks: list[str] = [self.doc_chunks[idx] for idx in indices[0]]
        return retrieved_chunks


sql_doc_retriever: RAG = RAG("sql_doc.txt")
sql_doc_retriever.SWITCH: bool = False  # TODO: RAG 功能稍后集成.


def most_representative_of(
    get_response: Callable[[], str],
    *,
    loop: int = 7,
) -> str:
    """从多个来自 AI 的回答中, 挑出最普遍的那个."""

    assert loop >= 1

    # 并发地获取多个来自 AI 的回答.
    with concurrent.futures.ThreadPoolExecutor(max_workers=loop) as executor:
        future_responses = {executor.submit(get_response) for _ in range(loop)}
        responses: list[str] = [
            future.result()
            for future in concurrent.futures.as_completed(future_responses)
        ]
    logger.info(f"\033[32m候选回答\033[0m {responses=!s}\n")

    # 将文本转为 TF-IDF 向量.
    tfidf_matrix = sklearn.feature_extraction.text.TfidfVectorizer().fit_transform(
        responses
    )

    # 计算余弦相似度矩阵.
    sim_matrix = sklearn.metrics.pairwise.cosine_similarity(tfidf_matrix)

    # 计算每个文本与其他文本的平均相似度 (不包括自身).
    avg_similarities = []
    for i in range(len(responses)):
        # 去除自身相似度 (即 1.0), 除以其它文本个数.
        avg_sim = (
            (numpy.sum(sim_matrix[i]) - 1) / (len(responses) - 1)
            if len(responses) > 1
            else 1.0
        )
        avg_similarities.append(avg_sim)

    # 找出平均相似度最高的文本.
    best_index = int(numpy.argmax(avg_similarities))
    return responses[best_index]


def get_db_info(
    *,
    ignore: frozenset[str] = frozenset(),
) -> tuple[int, str]:
    """用自然语言描述数据库中所有表的表名与列名.
    `IGNORE` 中的表会被当作 **完全不存在**.

    返回的元组的第一项表示数据库中表的数量;
    第二项是描述, 有两种说法:
      - “目前 数据库里 没有表.”
      - “目前 数据库里 有这些表 (字段名按顺序包含在表名后的圆括号中): A (b), C (d, e).”
    """

    tables: list[str] = [
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_master;").fetchall()
        if row["name"] not in ignore
    ]

    fields_tables: dict[str, list[str]] = {}
    for table in tables:
        fields: list[str] = [
            row["name"] for row in db.execute(f"PRAGMA table_info({table});").fetchall()
        ]
        fields_tables[table] = fields

    info: str = ", ".join(
        table + " (" + ", ".join(fields) + ")"
        for table, fields in fields_tables.items()
    )  # 形如 “A (b), C (d, e)”

    sentence: str = (
        "目前 数据库里 "
        + (
            f"有这些表 (字段名按顺序包含在表名后的圆括号中): {info}"
            if info
            else "没有表"
        )
        + "."
    )

    logger.info(f"\033[32m数据库信息\033[0m {sentence=!s}\n")
    return len(tables), sentence


def get_background(user_query: str) -> str:
    """根据 `USER_QUERY`, 提取出数据库中 **相关** 的元数据.

    返回可以直接嵌入到 prompt 中的文本段, 作为背景信息.
    """

    num_tables, db_info = get_db_info()

    if num_tables == 0:
        # 这种情况下, 直接返回 “目前 数据库里 没有表.”.
        return db_info

    all_tables: list[str] = [
        row["name"] for row in db.execute("SELECT name FROM sqlite_master;")
    ]

    def get_unrelated_tables() -> list[str]:
        """返回排好序的无关表名."""
        response: str = LLM.get_ai().get_response(
            messages=[
                {
                    "role": "system",
                    "content": f"""
你是一名信息检索员, 负责检索出数据库中与用户请求相关的表.
{db_info}

你的任务是:
分析用户的输入,
用 JSON Array, 必须形如:

```json
["table_name_1", "table_name_2", ...]
```

这样的格式, 列出你认为与用户的请求 *可能有关* 的 table 的名字 (哪怕是半毛钱关系).

注意: 用户 使用 *自然语言* 发起 数据库 查询请求.
                    """.strip(),
                },
                {"role": "user", "content": user_query},
            ],
            response_format={"type": "json_object"},
        )

        # 通义千问's bug:
        if response.startswith("```"):
            response = "\n".join(response.splitlines()[1:]).split("```")[1]
        logger.info(f"\033[32m相关表\033[0m {response=!r}")

        match (related_tables := json.loads(response)).__class__.__name__:
            case "list":
                pass
            case "dict" if "tables" in related_tables:
                related_tables = related_tables["tables"]
        logger.info(f"\033[32m相关表\033[0m {related_tables=!s}")
        # AI 可能不小心把字段名也包含进去了, 我们手动一个一个删掉:
        for i, table in enumerate(related_tables):
            related_tables[i] = table.strip().split()[0]
        # 此时 `related_tables` 形如 ["a", "b"].

        unrelated_tables: list[str] = [
            table for table in all_tables if table not in related_tables
        ]

        # 按字典序排序, 这样更可能出现相同的前缀.
        return sorted(unrelated_tables)

    unrelated_tables_candidates: list[tuple[str, list[str]]] = []
    unrelated_tables: str = most_representative_of(
        lambda: (
            unrelated_tables_candidates.append(
                (str(unrelated_tables := get_unrelated_tables()), unrelated_tables)
            ),
            "无关表: " + str(unrelated_tables),
        )[-1]
    ).lstrip("无关表: ")
    tables_to_ignore: list[str] = dict(unrelated_tables_candidates)[unrelated_tables]
    logger.info(f"\033[32m要忽略的表\033[0m {tables_to_ignore=!s}\n")

    _, db_info = get_db_info(ignore=frozenset(tables_to_ignore))
    return db_info


def sql_valid_p(sql: str) -> bool:
    """检查 SQL 语句是否合法.

    此处保证 SQL 语句以 **分号结尾**.
    """

    if not sqlite3.complete_statement(sql):
        return False

    with db_lock:
        try:
            db.execute(sql)
        except (
            # 只有此类型的错误才属于 user error:
            sqlite3.OperationalError,
            # 出现了多条 SQL 语句:
            sqlite3.ProgrammingError,
        ):
            return False
        else:
            return True
        finally:
            db.rollback()


def gen_sql(
    prompt: Prompt,
    num_tries: int = 3,
) -> str:
    """根据 `PROMPT`, 生成 SQL 语句.

    `NUM_TRIES` 表示最多尝试生成多少次 SQL 语句.

    如果 `NUM_TRIES` 次都没有生成合法的 SQL 语句, 我们把报错信息返回给 AI.
    再让 AI 根据这个报错信息, 重新尝试至多 `NUM_TRIES` 次.

    所有尝试均失败, 则抛出异常.
    查看 `err.__notes__[0]` 可以看到 AI 对此的解读.
    """

    assert num_tries >= 1  # 但不应太大.

    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": """
你负责帮助用户 将 自然语言 的 查询请求 转译为 SQL (数据库使用 SQLite3).

注意:
- 如果是 `CREATE TABLE` 语句, 你 **绝不应该** 书写 字段的数据类型, 因为 SQLite 支持 flexible typing.
- 你只能 将 用户 的 请求 转译成 **一句** SQL 语句, 哪怕你认为应该用多条 SQL 语句.
                        """.strip(),
        },
        {"role": "user", "content": str(prompt)},
    ]
    # 循环, 直到 AI 生成的 SQL 语句是合法的.
    for _ in range(num_tries):
        response: str = LLM.get_ai("coder").get_response(
            messages=[
                *msgs,
                {"role": "assistant", "content": "```sql\n", "partial": True},
            ],
        )
        sql = response.split("```")[0].strip()

        # 确保 SQL 语句以分号结尾.
        if sql[-1] != ";":
            sql += ";"

        if sql_valid_p(sql):
            logger.info(f"\033[32mSQL\033[0m {sql=!s}")
            return sql

    # 重试了 `NUM_TRIES` 次还是失败了...

    # 把 AI 的回答加入到消息历史中.
    msgs.append({"role": "assistant", "content": sql})
    # 注意, 此处的 `sql` 是我们修正过的 SQL 语句.

    try:
        # 我们用最后一次出错的 SQL 语句再次运行, 一边查看究竟是什么错误.
        with db:
            db.execute(sql)
    except (sqlite3.OperationalError, sqlite3.ProgrammingError) as err:
        logger.error(f"\033[31mSQL 错误\033[0m {err=!s}")

        for _ in range(num_tries):
            response: str = LLM.get_ai("coder").get_response(
                messages=[
                    *msgs,
                    {
                        "role": "user",
                        "content": f"""
我在 SQLite3 中执行了你给出的 SQL 语句, 结果有如下报错:

{"\n".join(f"> {line}" for line in str(err).splitlines())}

请重新生成 SQL 语句.

注意, 新生成的 SQL *必须满足我最初的需求*:

{'\n'.join(f'> {line}' for line in prompt.query.splitlines())}
                        """.strip(),
                    },
                    {
                        "role": "assistant",
                        "content": "```sql\n",
                        "partial": True,
                    },
                ],
            )

            # 梅开二度:
            sql = response.split("```")[0].strip()
            # 确保 SQL 语句以分号结尾.
            if sql[-1] != ";":
                sql += ";"
            if sql_valid_p(sql):
                return sql

        # 在我们把报错信息反馈给 AI, 让他重试了 `NUM_TRIES` 次之后还是失败了...

        # 很有可能是用户原本的查询请求压根就不合理, 让我们看看 AI 怎么说.
        reason: str = LLM.get_ai().get_response(
            messages=[
                *msgs,
                {
                    "role": "user",
                    "content": f"""
我在 SQLite3 中执行了你给出的 SQL 语句, 结果有如下报错:

{"\n".join(f"> {line}" for line in str(err).splitlines())}

哪里出了问题呢?
                        """.strip(),
                },
                {
                    "role": "assistant",
                    "content": """
你 原本 用 自然语言 提出 的 查询请求 很可能并不合理.
而我只是 如实地 将 你的请求 转译为 SQL 语句, SQL 语句本身不太可能出错.

鉴于 你对 SQL 知之甚少,
我将 完全避免 SQL 或 RDBMS 相关的 专业术语, 也不会提及刚才的 SQL 语句.

以下是 我 针对 你的最初的需求 进行 分析 所得到的 诊断:
                        """.strip()
                    + "\n\n",
                    "partial": True,
                },
            ],
        )

        err.add_note(reason)
        raise


def polish(query: str) -> str:
    """结合数据库的信息, 修正用户的请求.

    该函数会开启长对话.
    """

    msgs: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"""
你是一位 SQLite3 数据库管理人员.
{get_db_info()}

用户会用自然语言请求查询 SQLite3 数据库.
但你从不正面回应用户查询数据库的请求,
你的任务只是搞清楚用户的需求是什么, 偶尔给他提供一些数据库的背景信息.

在你无法理解用户的究竟想干什么、或认为该请求在查询数据库的语境下不成立时,
就你不理解的地方, 继续询问用户;
否则, 回答 “Understood.” 这句话, 不能多也不能少.

注意:
我们使用的 SQLite 支持 flexible typing 特性, 因此绝不需要指定数据类型.
""".strip(),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    # 一直询问直到 AI 说自己已经理解了.
    while True:
        # 获取 AI 的回答, 并立即加入到 消息历史 中去.
        # 如果连续五次都 understood, 则确实理解了.
        for _ in range(3):
            response = LLM.get_ai("chatter").get_response(messages=msgs)
            if "understood" not in response.lower():
                break
        else:
            break
        msgs.append({"role": "assistant", "content": response})

        # AI 没能理解, 于是 AI 提问.
        print("\n\033[94mAI:\033[0m " + response + "\n")

        # 用户回答.  (忽略任何空白输入.)
        while (user_input := input("\033[95m你:\033[0m ")).strip() == "":
            continue
        print()

        msgs.append({"role": "user", "content": user_input})

    logger.info(f"\033[32mAI 对用户需求的评价\033[0m {msgs[-1]['content']=!s}\n")
    # AI 总算是理解了用户的请求, 接下来我们让 AI 假装成用户, 并返回他的请求.
    msgs.append(
        {
            "role": "system",
            "content": """
你的目标其实是搞清楚用户的需求, 然后代替他向另一位的 AI Agent 提出请求.
鉴于你现在已经理解了用户的请求, 那么就请你假装成用户, 然后提出问题吧!

(注意: 你不被允许提及数据库的背景信息, 即数据库里有哪些表.)
            """.strip(),
        }
    )
    polished_query: str = most_representative_of(
        lambda: LLM.get_ai("chatter")
        .get_response(
            messages=[
                *msgs,
                {
                    "role": "assistant",
                    "content": "好的, 下面是我模仿用户, 给出的自然语言版查询请求:\n“",
                    "partial": True,
                },
            ],
        )
        .rstrip("”")  # 此时回答的格式是 `[“]...”`, 我们需要去掉.
    )

    logger.info(f"\033[32m润色后的请求\033[0m {polished_query=!s}\n")
    return polished_query


def get_sql(
    prompt: Prompt,
    num_tries: int = 7,
) -> str:
    """根据 `PROMPT`, 生成 SQL 语句.

    该函数会尝试生成很多次 SQL 语句.
    如果其中大部分次数是成功的, 就从成功的部分中选一个最具代表性的返回;
    否则, 从失败原因中找出最具代表性的作为异常抛出.
    """

    assert num_tries % 2 == 1
    # 保证有奇数个结果, 这样就不会出现平局.

    sqls: list[str] = []
    errors: list[sqlite3.OperationalError | sqlite3.ProgrammingError] = []

    def gen_sql_noexcept() -> None:
        try:
            sql = gen_sql(prompt)
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as err:
            errors.append(err)
            return

        # 格式化 SQL 语句, 消除不同 LLM 生成的 SQL 语句的格式差异:
        sql = sqlparse.format(
            sql,
            keyword_case="upper",
            strip_comments=False,
            indent_tabs=False,
            indent_width=2,
            compact=True,
        ).strip()
        sqls.append(sql)

    # 并发地获取多个来自 AI 的回答.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tries) as executor:
        futures = {executor.submit(gen_sql_noexcept) for _ in range(num_tries)}
        concurrent.futures.as_completed(futures)

    if len(sqls) > len(errors):
        sql = iter(sqls)
        return most_representative_of(
            lambda: next(sql),
            loop=len(sqls),
        )
    else:
        err = iter(errors)
        err_s: str = most_representative_of(
            lambda: str(next(err)),
            loop=len(errors),
        )
        for err in errors:
            if str(err) == err_s:
                raise err


def print_res(res: Iterable[dict[str, Any]]) -> None:
    """展示查询结果.

    如果数据是
      - 标量: 直接打印.
      - 列表: 先打印列名, 再打印各行数据.
      - 二维表格: 用 Web 页面展示.
    """

    res: list[dict[str, Any]] = list(res)

    height: int = len(res)
    print(f"一共 {height} 条记录.")

    # 没有数据, 直接返回.
    if height == 0:
        return

    match height, width := len(res[0]):
        # 标量:
        case 1, 1:
            print(f"结果: {next(iter(res[0].values()))}")
        # 列表:
        case _, 1:
            field: str = next(iter(res[0]))
            print(str({field: [row[field] for row in res]})[1:-1])
        # 二维表格:
        case _, _:
            fields: tuple[str, ...] = tuple(res[0])
            rows: list[tuple] = [tuple(row.values()) for row in res]

            # 立即用 Web 弹出窗口进行展示:

            # 将每行数据转置为每列数据:
            columns = list(zip(*rows))
            # 创建表格
            figure = plotly.graph_objects.Figure(
                data=[
                    plotly.graph_objects.Table(
                        header=dict(
                            values=list(fields),
                            fill_color="paleturquoise",
                            align="center",
                            font=dict(color="black", size=24),
                            height=38,
                        ),
                        cells=dict(
                            values=columns,
                            fill_color="lavender",
                            align="center",
                            font=dict(color="black", size=24),
                            height=38,
                        ),
                    )
                ]
            )
            # 优化布局:
            figure.update_layout(
                title="查询结果",
                margin=dict(l=20, r=20, t=40, b=40),
            )
            # 显示图表:
            figure.show()


def print_updated_then_confirm(
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> bool:
    """展示更新前后的涉及的记录, 并询问用户是否继续.

    1. 在命令行提示即将打开 Web 展示数据.
    2. 然后打开 Web, 左侧显示旧数据, 右侧显示新数据.
    3. 命令行询问用户是否确认要 UPDATE 表格.
    """

    assert old_records and new_records

    print("即将打开 Web 页面, 向您展示更新前后的数据", end="")
    for _ in range(3):
        print(".", end="", flush=True)
        time.sleep(1)
    print()

    # 两个表格的列名:
    fields: tuple[str, ...] = tuple(old_records[0])

    # 旧表格的数据:
    old_rows: list[tuple] = sorted(tuple(row.values()) for row in old_records)
    # 新表格的数据:
    new_rows: list[tuple] = sorted(tuple(row.values()) for row in new_records)
    # 上下拼接:
    rows: list[tuple] = old_rows + [tuple("* 更新后 *" for _ in fields)] + new_rows

    # TODO: `更新后` 这一行的颜色要改变.

    # 将每行数据转置为每列数据:
    columns: list[tuple] = list(zip(*rows))
    # 创建表格:
    figure = plotly.graph_objects.Figure(
        data=[
            plotly.graph_objects.Table(
                header=dict(
                    values=list(fields),
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(color="black", size=24),
                    height=38,
                ),
                cells=dict(
                    values=columns,
                    fill_color="lavender",
                    align="center",
                    font=dict(color="black", size=24),
                    height=38,
                ),
            )
        ]
    )
    # 优化布局:
    figure.update_layout(
        title="更新前后对照",
        margin=dict(l=20, r=20, t=40, b=40),
    )
    # 显示图表:
    figure.show()

    return "y" == input("确定要更新吗?  (Y/N): ").lower()


def main():
    try:
        print(
            "\n********************************** 新一轮会话 **********************************"
        )

        # 获取用户的初始请求.  (忽略任何空白输入.)
        while (prompt := Prompt(input("\033[95m你:\033[0m ").strip())) == "":
            continue
        print()

        # 询问用户, 直到我们搞清楚他究竟想干啥.
        prompt.query = polish(prompt.query)

        # 最终给到 AI 的查询请求.  这包含裁剪过的数据库元数据, 以及 AI 润色过的请求.
        prompt.background.append(get_background(prompt.query))
        # 使用 RAG 获取 SQL 相关的知识.
        if sql_doc_retriever.SWITCH:
            prompt.knowledge += sql_doc_retriever.retrieve_context(prompt.query, 3)
        logger.info(f"\033[32m正式提问\033[0m {prompt=!s}\n")

        try:
            sql: str = get_sql(prompt)
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as err:
            print(
                f"""
                \033[31m{err}\033[0m: {err}
\033[94mAI:\033[0m  您的请求可能不合理, 导致难以实现.  请听我解释...

{'\n'.join(f"    {line}" for line in err.__notes__[0].splitlines())}

您需要重新提出请求.
""".strip()
            )
            return

        # 高级用户可能想要自己输入 SQL 语句:
        if user_input_sql := input(
            f"""
这是即将执行的 SQL 语句:

\033[3m{
    '\n'.join(f"    {line}" for line in sql.splitlines())
}\033[0m

你可手动输入新的 SQL 语句替换它 (直接回车则使用 AI 生成的语句)
: """
        ).strip():
            while not sqlite3.complete_statement(user_input_sql):
                user_input_sql += "\n" + input(": ")
            else:
                try:
                    db.execute(user_input_sql)
                except sqlite3.Error as err:
                    print(err)
                    return
                finally:
                    db.rollback()
            sql = user_input_sql
        print()

        match sql.split()[0].upper():
            # 如果是查询语句, 我们直接打印.
            case "SELECT":
                with db:
                    res = db.execute(sql).fetchall()
                print_res(res)
            # 插入操作, 直接插入即可.
            case "INSERT":
                with db:
                    db.execute(sql)
            case "DELETE":
                # 先获取即将删除的行:
                sql = sql[:-1] + " RETURNING *;"
                rows_to_del: list[dict[str, Any]] = db.execute(sql).fetchall()

                if (
                    "y"
                    != input(
                        f"""以下是即将被删除的记录:

{"\n".join(str(row) for row in rows_to_del)}

是否继续?  (Y/N): """
                    )
                    .strip()
                    .lower()
                ):
                    print("已取消.")
                    return
                else:
                    with db:
                        db.execute(sql)
            # 更新前, 先获取即将被更新的行.
            # 然后展示更新后会变成什么样.
            case "UPDATE":
                # 我们用 diff SQL dump 的方式来获取即将被更新的行.

                old_dump: set[str] = {*db.iterdump()}
                logger.info(f"\033[32m旧数据\033[0m {old_dump=!s}")

                db.execute(sql)
                new_dump: set[str] = {*db.iterdump()}
                db.rollback()
                logger.info(f"\033[32m新数据\033[0m {new_dump=!s}")

                common_lines: set[str] = old_dump & new_dump
                old_dump -= common_lines
                new_dump -= common_lines

                if len(old_dump) == 0:
                    assert len(new_dump) == 0
                    print("没有记录被更新.")
                    return

                updated_table: str = re.fullmatch(
                    r'INSERT INTO "([^"]+)" VALUES\((?:.|\n)*\);',
                    next(iter(old_dump)),
                )[1]
                logger.info(f"\033[32m更新的表\033[0m {updated_table=!s}")
                # 查找创建表的 SQL 语句:
                for line in common_lines:
                    if re.fullmatch(
                        rf"CREATE TABLE {updated_table} \((?:.|\n)*\);",
                        line,
                    ):
                        creating_sql: str = line
                        break
                # 创建一个 in-memory 数据库:
                with contextlib.closing(
                    sqlite3.connect(
                        ":memory:",
                        autocommit=False,
                        check_same_thread=False,
                    )
                ) as tmp_db:
                    tmp_db.row_factory = db.row_factory
                    # 创建刚刚被更新的表:
                    with tmp_db:
                        tmp_db.execute(creating_sql)
                    # 插入旧数据:
                    for line in old_dump:
                        tmp_db.execute(line)
                    old_records: list[dict[str, Any]] = tmp_db.execute(
                        f'SELECT * FROM "{updated_table}";'
                    ).fetchall()
                    tmp_db.rollback()
                    # 插入新数据:
                    for line in new_dump:
                        tmp_db.execute(line)
                    new_records: list[dict[str, Any]] = tmp_db.execute(
                        f'SELECT * FROM "{updated_table}";'
                    ).fetchall()

                if not print_updated_then_confirm(old_records, new_records):
                    print("已取消.")
                    return
                else:
                    with db:
                        db.execute(sql)

            # 创建表, 直接创建即可.
            case "CREATE":
                with db:
                    db.execute(sql)
            # 删除表, 直接删除即可.
            case "DROP":
                with db:
                    db.execute(sql)
            # 修改表, 直接修改即可.
            case "ALTER":
                with db:
                    db.execute(sql)
            # 未知操作, 直接执行
            case _:
                with db:
                    db.execute(sql)
        print()

    # 用户刻意输入了 `^Z`, 以开启新一轮会话.
    except EOFError:
        return


if __name__ == "__main__":
    while True:
        main()
