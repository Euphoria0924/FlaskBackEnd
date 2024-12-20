import json
import threading
import time
from datetime import datetime

import pytz

from data_model import Session, Task
from train import train

background_worker_thread = None
def background_worker_train():
    """
    后台线程，定期扫描任务表并处理 PENDING 状态的任务
    """
    while True:
        session = Session()
        try:
            # 查询 PENDING 状态的任务
            pending_tasks = session.query(Task).filter_by(status="PENDING").all()
            if len(pending_tasks) >0 :
                print(f"Pending tasks found: {len(pending_tasks)}")
            for task in pending_tasks:
                # 更新任务状态为 IN_PROGRESS
                task.status = "IN_PROGRESS"
                try:
                    task.updated_at = datetime.utcnow()
                    session.commit()
                    # 解析 request_json
                    request_data = json.loads(task.request_json)
                    request_lr, request_epoch, request_model, request_dataset = request_data['lr'], request_data[
                        'epoch'], request_data['model'], request_data['dataset']
                    # 执行训练任务
                    result = train(request_lr, request_epoch, request_model, request_dataset,task.id)
                    # 更新任务状态为 COMPLETED，并保存结果
                    task.status = "COMPLETED"
                    task.response_json = json.dumps(result)
                except Exception as e:
                    # 如果训练失败，更新任务状态为 FAILED
                    task.status = "FAILED"
                    task.response_json = json.dumps({"error": str(e)})
                    print(f"Task {task.id} failed: {str(e)}")
                finally:
                    # 无论是否发生异常，确保提交更新
                    if task.status == "IN_PROGRESS":  # 检查是否仍为 IN_PROGRESS
                        task.status = "FAILED"
                        task.response_json = json.dumps({"error": "Unexpected termination"})
                    china_timezone = pytz.timezone('Asia/Shanghai')
                    task.updated_at = datetime.now(china_timezone)
                    session.commit()

                # 提交更新
                china_timezone = pytz.timezone('Asia/Shanghai')
                task.updated_at = datetime.now(china_timezone)
                session.commit()
        except Exception as e:
            print(f"Background worker error: {str(e)}")
        finally:
            session.close()
        # 等待一段时间后再次扫描
        time.sleep(5)

def start_background_worker():
    """
    启动后台任务线程（只会启动一次）
    """
    global background_worker_thread
    if background_worker_thread is None:
        # 创建并启动线程
        background_worker_thread = threading.Thread(target=background_worker_train, daemon=True)
        background_worker_thread.start()
        print("Background worker thread started.")