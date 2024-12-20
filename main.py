import os
import threading

import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import background_worker
from background_worker import background_worker_train, start_background_worker
from csv_process import csv_process
from db_manipulate import create_task, show_task_result
from train import train

app = Flask(__name__)
CORS(app, resources=r'/*')  # 允许所有路由的跨域请求@app.route("/")
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 如果目录不存在，创建目录
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/generate_task', methods=['POST'])
def task_generate():
    """
    接收模型训练参数的接口
    参数：
        - lr (float): 学习率
        - epoch (int): 训练轮数
        - model_name (str): 模型名称
    返回：
        - JSON 响应，包含输入参数和简单的反馈消息
    """
    request_dict = request.get_json()
    task_id = create_task(request_dict)
    return jsonify({"id": task_id, "message": "Task created successfully!"}),200


@app.route('/show_result')
def task_result():
    task_id = request.args.get('task_id')  # 如果没有传递 'id' 参数，默认返回 None
    if not task_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400  # 返回 400 错误
    return show_task_result(task_id)




@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """
    上传 CSV 文件并读取内容
    """
    # 检查是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # 检查是否是 CSV 文件
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

    # 保存文件到 UPLOAD_FOLDER
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    try:
        # 读取 CSV 内容
        df_example = csv_process(file_path)
        # 转换 DataFrame 为 JSON
        return jsonify(df_example.to_json(orient='records')), 200
    except Exception as e:
        return jsonify({'error': f'Failed to process CSV file: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def test():
    """
    接收模型训练参数的接口
    参数：
        - lr (float): 学习率
        - epoch (int): 训练轮数
        - model_name (str): 模型名称
    返回：
        - JSON 响应，包含输入参数和简单的反馈消息
    """
    request_dict = request.get_json()
    train(request_dict['lr'], request_dict['epoch'], request_dict['model'], request_dict['dataset'],233)
    return jsonify({"message": "ok!"}),200


if __name__ == "__main__":
    # 启动后台线程
    start_background_worker()
    app.run(debug=True)

