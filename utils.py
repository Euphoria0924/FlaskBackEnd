import numpy as np
# lookback表示观察的跨度
def split_data(feature, target, lookback):
    # 将股票数据转换为 numpy 数组
    data_raw = feature
    target_raw = target
    data = []
    target = []

    # 迭代数据，根据 lookback 参数生成输入序列
    # lookback 参数定义了要回溯的时间步长
    for index in range(len(data_raw) - lookback):
        # 从原始数据中截取从当前索引开始的 lookback 长度的数据
        data.append(data_raw[index: index + lookback])
        target.append(target_raw[index: index + lookback])

    # 将列表转换为 numpy 数组
    data = np.array(data)
    target = np.array(target)
    print(data.shape)
    print(target.shape)
    # 计算测试集的大小，这里取数据总量的 20%
    test_set_size = int(np.round(0.2 * data.shape[0]))

    # 计算训练集的大小
    train_set_size = data.shape[0] - test_set_size

    # 分割数据为训练集和测试集
    # x_train 和 x_test 包含除了最后一列外的所有数据
    # y_train 和 y_test 包含最后一列数据
    x_train = data[:train_set_size, :-1, :]
    y_train = target[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = target[train_set_size:, -1, :]

    # 返回分割后的训练数据和测试数据
    return [x_train, y_train, x_test, y_test]


def column_indices(df, colnames):
    """返回列名对应的索引列表"""
    return [df.columns.get_loc(c) for c in colnames if c in df.columns]


def split_data2(feature, target, lookback):
    # 将股票数据转换为 numpy 数组
    data_raw = feature
    target_raw = target
    data = []
    target = []

    # 迭代数据，根据 lookback 参数生成输入序列
    # lookback 参数定义了要回溯的时间步长
    for index in range(len(data_raw) - lookback):
        # 这里我们仅使用前 lookback 天的数据来预测目标
        data.append(data_raw[index: index + lookback])  # 数据从第 index 天到 index + lookback 天
        target.append(target_raw[index + lookback - 1])  # 预测的是第 index + lookback 天的目标

    # 将列表转换为 numpy 数组
    data = np.array(data)
    target = np.array(target)
    print(data.shape)
    print(target.shape)

    # 计算测试集的大小，这里取数据总量的 20%
    test_set_size = int(np.round(0.2 * data.shape[0]))

    # 计算训练集的大小
    train_set_size = data.shape[0] - test_set_size

    # 分割数据为训练集和测试集
    # x_train 和 x_test 包含除了最后一列外的所有数据
    # y_train 和 y_test 包含最后一列数据
    x_train = data[:train_set_size, :, :]  # x_train 是前 n 天的数据
    y_train = target[:train_set_size]  # y_train 是第 n 天的目标
    x_test = data[train_set_size:, :, :]  # x_test 是前 n 天的数据
    y_test = target[train_set_size:]  # y_test 是第 n 天的目标

    # 返回分割后的训练数据和测试数据
    return [x_train, y_train, x_test, y_test]


def split_data_multistep(feature, target, lookback, future_steps):
    """
    时间序列多步预测数据分割。
    :param feature: 输入特征数组（形状为 [样本数, 特征数]）
    :param target: 目标值数组（形状为 [样本数, 目标数]）
    :param lookback: 时间窗口大小（用于输入特征）
    :param future_steps: 未来预测的时间步数
    :return: x_train, y_train, x_test, y_test
    """
    data = []
    target_data = []

    # 遍历数据，根据 lookback 和 future_steps 生成样本
    for index in range(len(feature) - lookback - future_steps + 1):
        # 生成时间窗口
        data.append(feature[index: index + lookback])
        # 目标是从当前窗口末端到未来的 future_steps 行
        target_data.append(target[index + lookback: index + lookback + future_steps])

    # 转为 numpy 数组
    data = np.array(data)
    target_data = np.array(target_data)
    print("Feature shape (x):", data.shape)  # (样本数, 时间窗口, 特征数)
    print("Target shape (y):", target_data.shape)  # (样本数, 未来步数, 目标数)

    # 分割数据集
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size]
    y_train = target_data[:train_set_size]
    x_test = data[train_set_size:]
    y_test = target_data[train_set_size:]

    return [x_train, y_train, x_test, y_test]
