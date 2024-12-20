import json
from datetime import datetime, time
import pandas as pd
import pytz
from flask import jsonify, request


from data_model import Session, Task, engine



def fetch_table_as_df(table_name):
    """
    根据表名查询 MySQL 表，并返回 Pandas DataFrame

    参数:
        - table_name (str): 表名

    返回:
        - Pandas DataFrame: 查询结果
    """
    # 校验表名是否合法
    # if not table_name or not table_name.isidentifier():
    #     raise ValueError("Invalid table name. The table name must be a valid identifier.")

    # 获取数据库引擎

    # 查询表数据并转换为 DataFrame
    query = f"SELECT * FROM `{table_name}`"  # 使用反引号防止表名中出现特殊字符
    df = pd.read_sql(query, engine)
    return df


# SQLAlchemy 基础配置
def create_task(task_dict):
    """
    保存任务到任务表

    参数:
        - task_data (dict): 包含任务信息的字典

    返回:
        - task_id (int): 保存的任务 ID
    """
    # 将请求数据转为 JSON
    request_json = json.dumps(task_dict)
    china_timezone = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(china_timezone)
    # 创建数据库会话
    session = Session()
    try:
        # 创建任务实例
        new_task = Task(
            request_json=request_json,
            status="PENDING",
            created_at=current_time,
            updated_at=current_time
        )
        session.add(new_task)
        session.commit()
        task_id = new_task.id  # 获取插入的任务 ID
        return task_id
    except Exception as e:
        session.rollback()  # 回滚事务
        raise e
    finally:
        session.close()



def  search_task_result(task_id):
    """
    根据任务 ID 查询任务结果

    参数:
        - task_id (int): 任务 ID

    返回:
        - result (dict): 查询结果
    """
    # 创建数据库会话
    session = Session()
    try:
        # 查询任务结果
        task = session.query(Task).get(task_id)
        if task:
            response_dict = json.loads(task.response_json) if task.response_json else {}
            request_dict = json.loads(task.request_json) if task.request_json else {}
            result = {
                "id": task.id,
                "status": task.status,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "lr": request_dict.get('lr', 'None'),
                "epoch": request_dict.get('epoch', 'None'),
                "model": request_dict.get('model', 'None'),
                "dataset": request_dict.get('dataset', 'None'),
                "RMSE": response_dict.get('Test RMSE', 'None')

            }
            return jsonify(result)
        else:
            return {"error": "Task not found."}
    except Exception as e:
        session.rollback()  # 回滚事务
        raise e
    finally:
        session.close()


def show_all_tasks():
    session = Session()
    try:
        tasks = session.query(Task).all()
        task_list = []
        for task in tasks:
            response_dict = json.loads(task.response_json) if task.response_json else {}
            request_dict = json.loads(task.request_json) if task.request_json else {}
            task_list.append({
                "id": task.id,
                "status": task.status,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "lr": request_dict.get('lr', 'None'),
                "epoch": request_dict.get('epoch', 'None'),
                "model": request_dict.get('model', 'None'),
                "dataset": request_dict.get('dataset', 'None'),
                "RMSE": response_dict.get('Test RMSE', 'None')

            })
        return jsonify(task_list)
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()