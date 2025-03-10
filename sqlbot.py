#! python3.13
# -*- coding: utf-8-unix; -*-

# 标准库
import sqlite3
import logging
import json
import concurrent.futures
import threading
import os
from typing import (
    Any,
    Callable,
)

# 第三方库, 记得 pip install 哦~
import openai
import numpy
import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import sqlparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

db = sqlite3.connect(
    "eg.sqlite3",
    autocommit=False,
    # 我们会在多线程中访问数据库, 因此需要关闭检查线程是否相同的功能.
    check_same_thread=False,
)
db_lock = threading.Lock()  # 简单起见, 无论读写, 我们都默认加锁.
db.row_factory = lambda cursor, row: {
    field: value
    for field, value in zip(
        [field_info[0] for field_info in cursor.description],
        row,
    )
}

llm_client = openai.OpenAI(
    api_key=os.getenv("TongYiQianWen_API_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
llm_name = "qwen2.5-14b-instruct-1m"


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
      - “已知 目前 数据库里 没有表.”
      - “已知 目前 数据库里 有这些表 (字段名按顺序包含在表名后的圆括号中): A (b), C (d, e).”
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
        "已知 目前 数据库里 "
        + (
            f"有这些表 (字段名按顺序包含在表名后的圆括号中): {info}"
            if info
            else "没有表"
        )
        + "."
    )

    logger.info(f"\033[32m数据库信息\033[0m {sentence=!s}\n")
    return len(tables), sentence


def gen_context(user_query: str) -> str:
    """根据 `USER_QUERY`, 提取出数据库中 **相关** 的元数据.

    返回可以直接嵌入到 prompt 中的文本段, 作为背景信息.
    """

    num_tables, db_info = get_db_info()

    if num_tables == 0:
        # 这种情况下, 直接返回 “已知 目前 数据库里 没有表.”.
        return db_info

    all_tables: list[str] = db.execute("SELECT name FROM sqlite_master;").fetchall()

    def get_unrelated_tables() -> list[str]:
        """返回排好序的无关表名."""
        response: str = (
            llm_client.chat.completions.create(
                model=llm_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"""
你是一名信息检索员, 负责检索出数据库中与用户请求相关的表.
{db_info}

你的任务是:
分析用户的输入,
用 JSON Array 的形式 (必须形如 `["table_name_1", "table_name_2", ...]`),
列出你认为与用户的请求 *可能有关* 的 table 的名字.

注意: 用户 使用 自然语言 发起 数据库 查询请求.
                    """.strip(),
                    },
                    {"role": "user", "content": user_query},
                ],
                response_format={"type": "json_object"},
            )
            .choices[0]
            .message.content
        ).strip()

        # 通义千问's bug:
        if response.startswith("```"):
            response = "\n".join(response.splitlines()[1:]).split("```")[1]

        related_tables: list[str] = json.loads(response)
        for i, table in enumerate(related_tables):
            related_tables[i] = table.strip().split()[0]
        # 此时 `related_tables` 形如 ["a", "b"].

        unrelated_tables: list[str] = [
            table for table, in all_tables if table not in related_tables
        ]
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

    _, db_info = get_db_info(ignore=frozenset(unrelated_tables))
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
        except sqlite3.OperationalError:
            # 只有此类型的错误才属于 user error, 其它错误说明我们的代码有问题.
            return False
        else:
            return True
        finally:
            db.rollback()


def gen_sql(
    prompt: str,
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
- 如果是 CREATE 语句, 你 将 省略 数据类型, 因为 SQLite 支持 flexible typing.
                        """.strip(),
        },
        {"role": "user", "content": prompt},
    ]
    # 循环, 直到 AI 生成的 SQL 语句是合法的.
    for _ in range(num_tries):
        response: str = (
            llm_client.chat.completions.create(
                model=llm_name,
                messages=[
                    *msgs,
                    {"role": "assistant", "content": "```sql\n", "partial": True},
                ],
            )
            .choices[0]
            .message.content
        ).strip()
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
    except sqlite3.OperationalError as err:
        for _ in range(num_tries):
            response: str = (
                llm_client.chat.completions.create(
                    model=llm_name,
                    messages=[
                        *msgs,
                        {
                            "role": "user",
                            "content": f"""
我在 SQLite3 中执行了你给出的 SQL 语句, 结果有如下报错:

{"\n".join(f"> {line}" for line in str(err).splitlines())}

请重新生成 SQL 语句.
                        """.strip(),
                        },
                        {
                            "role": "assistant",
                            "content": "```sql\n",
                            "partial": True,
                        },
                    ],
                )
                .choices[0]
                .message.content
            ).strip()

            # 梅开二度:
            sql = response.split("```")[0].strip()
            # 确保 SQL 语句以分号结尾.
            if sql[-1] != ";":
                sql += ";"
            if sql_valid_p(sql):
                return sql

        # 在我们把报错信息反馈给 AI, 让他重试了 `NUM_TRIES` 次之后还是失败了...

        # 很有可能是用户原本的查询请求压根就不合理, 让我们看看 AI 怎么说.
        reason: str = (
            llm_client.chat.completions.create(
                model=llm_name,
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
            .choices[0]
            .message.content
        ).strip()

        logger.info(f"\033[32m需求无法实现的理由\033[0m {reason=!s}\n")
        err.add_note(response)

        raise


def polish(query: str) -> str:
    """结合数据库的信息, 修正用户的请求.

    该函数会开启长对话.
    """

    msgs: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"""
你是一位数据库管理人员.
{get_db_info()}

用户会用自然语言请求查询数据库.
但你从不正面回应用户查询数据库的请求, 你的任务只是搞清楚用户的需求是什么.

在你确认你能够理解用户的请求时, 回答 “Understood.” 这句话, 不能多也不能少;
否则, 就你不理解的地方, 继续询问用户.
""".strip(),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    # 一直询问直到 AI 说自己已经理解了.
    while "Understood." not in (
        # 获取 AI 的回答, 并立即加入到 消息历史 中去.
        response := msgs.__iadd__(
            [
                llm_client.chat.completions.create(
                    model=llm_name,
                    messages=msgs,
                )
                .choices[0]
                .message
            ]
        )[-1].content
    ):
        # AI 没能理解, 于是 AI 提问.
        print("\n\033[94mAI:\033[0m " + response + "\n")

        # 用户回答.  (忽略任何空白输入.)
        while (user_input := input("\033[95m你:\033[0m ")).strip() == "":
            continue
        print()

        msgs.append({"role": "user", "content": user_input})

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
        lambda: llm_client.chat.completions.create(
            model=llm_name,
            messages=[
                *msgs,
                {
                    "role": "assistant",
                    "content": "好的, 下面是我模仿用户, 给出的自然语言版查询请求:\n“",
                    "partial": True,
                },
            ],
        )
        .choices[0]
        .message.content.strip()
        .rstrip("”")  # 此时回答的格式是 `[“]...”`, 我们需要去掉.
    )

    logger.info(f"\033[32m润色后的请求\033[0m {polished_query=!s}\n")
    return polished_query


def get_sql(
    prompt: str,
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
    errors: list[sqlite3.OperationalError] = []

    def gen_sql_noexcept() -> None:
        try:
            return sqls.append(gen_sql(prompt))
        except sqlite3.OperationalError as err:
            return errors.append(err)

    # 并发地获取多个来自 AI 的回答.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tries) as executor:
        futures = {executor.submit(gen_sql_noexcept) for _ in range(num_tries)}
        concurrent.futures.as_completed(futures)

    if len(sqls) > len(errors):
        sql = iter(sqls)
        return most_representative_of(
            lambda: next(sql),
            loop=num_tries,
        )
    else:
        err = iter(errors)
        err_s: str = most_representative_of(
            lambda: str(next(err)),
            loop=num_tries,
        )
        for err in errors:
            if str(err) == err_s:
                raise err


while True:
    try:
        print(
            "********************************** 新一轮会话 **********************************"
        )

        # 获取用户的初始请求.  (忽略任何空白输入.)
        while (user_query := input("\033[95m你:\033[0m ")).strip() == "":
            continue
        print()

        # 询问用户, 直到我们搞清楚他究竟想干啥.
        user_query: str = polish(user_query)

        # 最终给到 AI 的查询请求.  这包含裁剪过的数据库元数据, 以及 AI 润色过的请求.
        prompt: str = gen_context(user_query) + "\n\n" + user_query
        logger.info(f"\033[32m正式提问\033[0m {prompt=!s}\n")

        sql: str = get_sql(prompt)
        # 格式化 SQL 语句:
        sql = sqlparse.format(
            sql,
            keyword_case="upper",
            strip_comments=False,
            indent_tabs=False,
            indent_width=2,
            compact=True,
        ).strip()
        logger.info(f"\033[32m格式化后的 SQL\033[0m {sql=!s}\n")
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
            sql = user_input_sql
        print()

        with db:
            res = db.execute(sql)
        for row in res:
            print(row)
        print()

    # 用户刻意输入了 `^Z`, 以开启新一轮会话.
    except EOFError:
        continue
