from sqlalchemy import create_engine, Column, String, Integer, Text, Enum, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# 数据库配置
DB_URL = "mysql+pymysql://root:123456@localhost:3306/drilldb"
engine = create_engine(DB_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Task 表模型
class Task(Base):
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)
    request_json = Column(Text, nullable=False)
    response_json = Column(Text, nullable=True)
    status = Column(Enum('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'), default='PENDING')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

# 初始化表
Base.metadata.create_all(engine)
