#! python3.13
# -*- coding: utf-8-unix; -*-

import openai, json, pathlib

client = openai.OpenAI(
    api_key="sk-b17d592c403047989ee008e4d712b404",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
model = "qwen2.5-1.5b-instruct"


def test_compl():
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "knock knock."},
            {"role": "assistant", "content": "Who's there?"},
            {"role": "user", "content": "Orange."},
        ],
    )
    print(completion.choices[0].message)


def test_json():
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'You extract email addresses into JSON data, for example, `{"email address":"user@example.org"}`',
            },
            {
                "role": "user",
                "content": "Feeling stuck? Send a message to help@mycompany.com.",
            },
        ],
        response_format={"type": "json_object"},
    )
    json_str = response.choices[0].message.content
    print(json.loads(json_str))


def test_prefix():  # 通义千问
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一名诗人。"},
            {"role": "user", "content": "为我写一句诗。"},
            {"role": "assistant", "content": "天青色等烟雨，", "partial": True},
        ],
    )
    print(response.choices[0].message.content)


def test_funcall():
    def get_weather(latitude, longitude, units):
        response = __import__("requests").get(
            f"https://api.open-meteo.com/v1/forecast?temperature_unit={units}&latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "根据经纬度获取当前气温",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "description": "纬度"},
                        "longitude": {"type": "number"},
                        "units": {
                            "type": "string",
                            "description": "温度单位",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["latitude", "longitude", "units"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]
    messages = [{"role": "user", "content": "上海市浦东新区当前的气温是多少?"}]
    messages.append(
        client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )
        .choices[0]
        .message
    )
    print(messages[-1].tool_calls)
    first_call = messages[-1].tool_calls[0]
    # 如果后面还有, 都要 call 一下并 append 到 messages.
    print(args := json.loads(first_call.function.arguments))
    print(
        result := {
            "role": "tool",
            "tool_call_id": first_call.id,
            "content": str(
                get_weather(args["latitude"], args["longitude"], args["units"])
            ),
        }
    )
    messages.append(result)
    print(
        client.chat.completions.create(model=model, messages=messages, tools=tools)
        .choices[0]
        .message.content
    )


def test_file():
    fobj = client.files.create(file=pathlib.Path("eg.openai.py"), purpose="file-extract")
    print(fobj)
    print(client.files.retrieve(file_id=fobj.id))
    print(client.files.list())
    print(
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "介绍上传的 Python 文件所提供的 API",
                }
            ],
        )
        .choices[0]
        .message.content
    )
    for f in client.files.list():
        client.files.delete(f.id)
