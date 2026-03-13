import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
app = Flask(__name__)

GPT_MODEL = "qwen-plus"

load_dotenv()
# 获取环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTHORIZATION_CODE = os.getenv("AUTHORIZATION_CODE")
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(
    max_results=5,
    tavily_api_key=TAVILY_API_KEY
)


client = OpenAI(

    api_key=OPENAI_API_KEY,

    base_url="https://api.zhizengzeng.com/v1",
)

# 全局变量存储对话历史
conversation_history = [
    {
            "role": "system",
            "content": """你叫贾维斯是一个AI助手。你需要与用户进行持续的多轮对话，直到用户明确表示想要结束对话。

对话规则：
1. 和你对话的对象名叫丹尼先生，你是他的高科技助手，请用类似于电影《钢铁侠》中的助手贾维斯科技的语气回答问题；
2. 如果丹尼先生没有提供目标列名，请提示他提供目标列名；
3. 如果丹尼先生给了你目标列，可能是不带有引号的，例如SibSp，请加上引号；
3. 如果用户希望你帮他分析目标列，请调用 analyze_data 工具；
4. 如果用户希望你帮他可视化目标列，请调用 draw 工具；
5. 如果用户希望你帮他预测目标列，请调用 predict 工具，并提示用户需要输入一个字典，示例如下：
    input_data = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Cabin": "C92",
        "Embarked": "S"
    }
4. 如果当用户询问实时信息、新闻、最新数据、人物近况、科技进展等内容时，应优先调用 web_search 工具。
5. 在每轮对话中，保持对话的连贯性，记住之前的对话内容;
6. 如果用户表达以下意图，请结束对话：
   - 明确说"再见"、"拜拜"、"结束对话"等告别语
   - 表达"我要走了"、"对话到此为止"等结束意图
   - 使用"exit"、"quit"等退出命令"""
    }
]


def analyze_data(target_col):
    """
    对于某一列数据，先判断是否为数值型，接着提供基本统计信息

    Args:
        target_col (str): 目标列名

    Returns:
        dict: 包含统计信息和数据类型的字典
    """
    stats = {}
    col_data = df[target_col]
    try:
        if col_data.dtype == 'object':
            # 统计字符串类型列中每个元素的出现次数和比例
            element_counts = Counter(col_data)
            element_counts = dict(element_counts)
            element_proportions = {element: count / len(col_data) for element, count in element_counts.items()}
            stats = {
                "element_counts": element_counts,
                "element_proportions": element_proportions,
                "data_type": str(col_data.dtype),
            }
            return stats
        elif col_data.dtype == 'bool':
            raise ValueError({"error": f"列 '{target_col}' 是布尔类型，不支持统计分析"})
        else:
            stats = {
                "mean": col_data.mean().item(),#转为python原生类型
                "median": col_data.median().item(),
                "std_dev": col_data.std().item(),
                "min": col_data.min().item(),
                "max": col_data.max().item(),
                "count": col_data.count().item(),
                "unique_values": col_data.nunique(),
                "data_type": str(col_data.dtype)
            }
            return stats
    except KeyError:
        raise ValueError(f"列 {target_col} 不存在")


def draw(target_col):
    """
    对目标列进行可视化

    Args:
        target_col (str): 目标列名


    Returns:
        str: 图片的base64编码
    """
    col_data = df[target_col]
    stats = analyze_data(target_col)
    #对于字符串类型的列，绘制饼图
    if stats["data_type"] == 'object':
        # 绘制饼图
        plt.figure(figsize=(8, 8))
        plt.pie(stats["element_proportions"].values(),
                labels=stats["element_proportions"].keys(),
                autopct='%1.1f%%', startangle=140)
        # 显示图例
        plt.legend()
        # 调整图例位置
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        #显示图表
        plt.title(f'Pie Chart of {target_col}')
        #将图片转换为base64
        image_base64 = fig_to_base64()
        return  image_base64
        # 对于数值型列，绘制直方图
    elif stats["data_type"] in ['int64', 'float64']:

        #若只有小于三个的值，绘制饼图
        if stats["unique_values"] <= 3:
            element_counts = Counter(col_data)
            element_counts = dict(element_counts)
            plt.figure(figsize=(8, 8))
            stats["element_proportions"] = {element: count / len(col_data) for element, count in element_counts.items()}
            plt.pie(stats["element_proportions"].values(),
                    labels=stats["element_proportions"].keys(),
                    autopct='%1.1f%%', startangle=140)
            # 显示图例
            plt.legend()
            # 调整图例位置
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            #显示图表
            plt.title(f'Pie Chart of {target_col}')
            #将图片转换为base64
            image_base64 = fig_to_base64()
            plt.close()

        else:
                # 绘制直方图
                plt.figure(figsize=(10, 6))
                plt.hist(col_data, bins=20, edgecolor='black')
                plt.title(f'Histogram of {target_col}')
                plt.xlabel(target_col)
                plt.ylabel('Frequency')
                #将图片转换为base64
                image_base64 = fig_to_base64()
        return image_base64
    elif stats["data_type"] == 'bool':
        raise ValueError({"error": f"列 '{target_col}' 是布尔类型，不支持可视化"})
    else:
        raise ValueError({"error": f"列 '{target_col}' 数据类型未知，不支持可视化"})

def train(df):
    #利用sklearn的随机森林模型对Survived进行预测
    # 初始化随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # 对数据进行预处理
    df1=df.copy()
    df1['Age'].fillna(df1['Age'].median(), inplace=True)
    df1['Embarked'].fillna(df1['Embarked'].mode()[0], inplace=True)
    # 转换性别为数值
    df1['Sex'] = df1['Sex'].map({'male': 0, 'female': 1})
    # 转换登船港口为数值
    df1['Embarked'] = df1['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    # 转换船舱号为数值
    cabin_map = {c:i for i,c in enumerate(df1['Cabin'].unique())}
    df1['Cabin'] = df1['Cabin'].map(cabin_map)
    # 删除非数值列
    df_model=df1.drop(columns=['PassengerId','Name','Ticket'])
    X = df_model.drop(columns=['Survived'])
    y = df_model['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    return model,model_score

def predict(input_data):
    """
    根据输入信息对Survived进行预测

    Args:
        input_data (dict): 包含乘客信息的字典，键为列名，值为对应的值

    Returns:
        str: 预测结果
    """
    global model
    # 输入数据预处理
    # 从输入数据中提取特征
    Pclass = input_data.get('Pclass', 0)
    Sex = input_data.get('Sex', 'male')
    Age = input_data.get('Age', 0.0)
    SibSp = input_data.get('SibSp', 0)
    Parch = input_data.get('Parch', 0)
    Fare = input_data.get('Fare', 0.0)
    Cabin = input_data.get('Cabin', '')
    Embarked = input_data.get('Embarked', 'S')
    # 转换船舱号为数值
    cabin_map = {c:i for i,c in enumerate(df['Cabin'].unique())}
    Cabin = cabin_map.get(Cabin,0)
    # 转换性别为数值
    Sex = 0 if Sex == 'male' else 1
    # 转换登船港口为数值
    Embarked = {'S': 0, 'C': 1, 'Q': 2}.get(Embarked, 0)

 
    # 对输入数据进行预测
    y_pred = model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked]])
    # 计算准确率
    if y_pred[0] == 1:
        return "Survived"
    else:
        return "Not Survived"

tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            # 因为获取当前时间无需输入参数，因此parameters为空字典
            "parameters": {},
        },
    },
    # 工具3 绘图
    {
        "type": "function",
        "function": {
            "name": "draw",
            "description": "Draw a plot based on the given data",   
            "parameters": {
                "type": "object",
                "properties": {
                    "target_col": {
                    "type": "string",
                    "description": "The target column name"
                }
                },
                "required": ["target_col"],
            },
        }
    },
    #数据总览
    {
    "type": "function",
    "function": {
        "name": "analyze_data",
        "description": "Provide a summary of the target column, including basic statistics and data types",
        "parameters": {
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "The target column name"
                }
            },
            "required": ["target_col"]
        }
    }
    },#搜索
    {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search real-time information from the internet using Tavily",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
    },#预测
    {
        "type": "function",
        "function": {
            "name": "predict",
            "description": "Predict whether a passenger would survive the Titanic disaster based on provided information",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {
                        "type": "object",
                        "description": "Input information for prediction"
                    }
                },
                "required": ["input_data"]
            }
        }
    }


]



def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"




def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e





@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history

    data = request.json
    user_message = data.get('message', '')

    # 检查用户是否想要结束对话
    if user_message.lower() in ['再见', '拜拜', '结束对话', 'exit', 'quit', '我要走了', '对话到此为止']:
        return jsonify({
            'response': '贾维斯一直待命中！',
            'end_conversation': True
        })

    conversation_history.append({"role": "user", "content": user_message})

    response = chat_completion_request(
        messages=conversation_history,
        tools=tools
    )
    message = response.choices[0].message
    print("===== FIRST RESPONSE =====")
    print(response.choices[0].message)
    print("==========================")
    if message.tool_calls:

        tool_call = message.tool_calls[0]
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        print("TOOL NAME:", fn_name)
        print("TOOL ARGS:", fn_args)
        if fn_name == "web_search":
            result = tavily_tool.invoke({"query": fn_args["query"]})
               #tool call写入history要符合openai的格式
               #assistant (tool_calls)
               #tool (result)
               #assistant (final answer)
               
            conversation_history.append({
                "role": "assistant",
                "tool_calls": message.tool_calls
            })

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

            final_response = chat_completion_request(
                messages=conversation_history
            )

            print("===== SECOND RESPONSE =====")
            print(final_response.choices[0].message)
            print("==========================")

            return jsonify({
                    'response': final_response.choices[0].message.content,
                })

        elif fn_name == "analyze_data":
            result = analyze_data(fn_args["target_col"])
            print(result)
            for k, v in result.items():
                print(k, type(v))
            try:    
                
                result = analyze_data(fn_args["target_col"])
                if result['data_type'] == 'object':
                    response = (
                        f"数据分析完成：\n"
                        f"各元素数量和比例: {result['element_counts']}\n"
                        f"各元素比例: {result['element_proportions']}\n"
                        f"数据类型: {result['data_type']}\n"
                    )

                else:
                    response = (
                        f"数据分析完成：\n"
                        f"数量: {result['count']}\n"
                        f"平均值: {result['mean']}\n"
                        f"中位数: {result['median']}\n"
                        f"标准差: {result['std_dev']}\n"
                        f"最大值: {result['max']}\n"
                        f"最小值: {result['min']}"
                        f"唯一值数量: {result['unique_values']}"
                        f"数据类型: {result['data_type']}"
                    )
                print(response)
                conversation_history.append({
                    "role": "assistant",
                    "tool_calls": message.tool_calls
                    })
                conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                final_response = chat_completion_request(
                        messages=conversation_history
                    )
                print("===== SECOND RESPONSE =====")
                print(final_response.choices[0].message)
                print("==========================")
                return jsonify({
                    'response': final_response.choices[0].message.content,
                    })          
            except Exception as e:  
                return jsonify({
                "response": f"数据分析出错：{str(e)}",
                "end_conversation": False
                })       
        elif fn_name == "draw":
            try:
                    target_col = fn_args["target_col"]
                    result = draw(target_col)
                    response = "Image generated successfully"
                    conversation_history.append({
                    "role": "assistant",
                    "tool_calls": message.tool_calls
                    })
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                    final_response = chat_completion_request(
                        messages=conversation_history
                    )
                    print("===== SECOND RESPONSE =====")
                    print(final_response.choices[0].message)
                    print("==========================")
                    return jsonify({
                        'response': final_response.choices[0].message.content,
                        'end_conversation': False,
                        'image': result
                    })
            except Exception as e:
                    return jsonify({
                        'response': f"可视化时出错：{str(e)}",
                        'end_conversation': False
                    })

        elif fn_name == "get_current_time":
            try:
                    now_time = get_current_time()
                    response = f"函数输出信息：{now_time}"
                    conversation_history.append({
                    "role": "assistant",
                    "tool_calls": message.tool_calls
                    })
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(now_time)
                    })
                    final_response = chat_completion_request(
                        messages=conversation_history
                    )
                    return jsonify({
                        'response': final_response.choices[0].message.content,
                        'end_conversation': False,
                    })

            except Exception as e:
                    print(f"查找时间出错：{e}")
                    conversation_history.append(
                        {"role": "assistant", "content": "抱歉，无法查找当前时间！"})
                    return jsonify(response)
        elif fn_name == "predict":
            try:
                result = predict(fn_args["input_data"])
                response = (
                    f"预测完成：\n"
                    f"预测结果: {result}\n"
                )
                conversation_history.append({
                    "role": "assistant",
                    "tool_calls": message.tool_calls
                    })
                conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                final_response = chat_completion_request(
                        messages=conversation_history
                    )
                return jsonify({
                    "response": final_response.choices[0].message.content,
                    "end_conversation": False
                }) 
            except Exception as e:  
                    return jsonify({
                "response": f"预测出错：{str(e)}",
                "end_conversation": False
                })       

    # 如果没有 tool call 才直接返回
    else:
        content = message.content
        conversation_history.append({"role": "assistant", "content": content})
        return jsonify({
            'response': content,
            'end_conversation': False
        })


def fig_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return img    

    


if __name__ == "__main__":
    # 读取titanic_cleaned.csv文件
    with open('10agentforanalysis\\titanic_cleaned.csv', 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    model,model_score=train(df)
    print(f"启动预测模型，准确率为: {model_score:.2f}")
    app.run(debug=True,port=8080)
