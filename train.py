import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
import torch

import math, time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

from db_manipulate import fetch_table_as_df
from model import LSTMAttention_ekan, LSTM
from utils import *


#形参设定
parser = argparse.ArgumentParser()
parser.add_argument('--input_features', type=list, default=['BIT DEPTH', 'HK LOAD', 'SPP','FLOW OUT','WOB','FLOW IN','TRQ','RPM','Size'])
parser.add_argument('--output_features', type=list, default=['ROP'])
parser.add_argument('--window_size', type=int, default = 15)
parser.add_argument('--train_test_ratio', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=30)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=2)
##kan
parser.add_argument('--grid_size', type=int, default=100, help='grid')
##TCN
parser.add_argument('--num_channels', type=list, default=[25, 50, 25])
parser.add_argument('--kernel_size', type=int, default=3)
##transformer
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--hidden_space', type=int, default=32)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args(args=[])
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(request_lr,request_epoch,request_model,request_dataset,task_id):
    request_lr = float(request_lr)
    request_epoch = int(request_epoch)
    save_dir = f"./model_source/{task_id}"
    os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    data = fetch_table_as_df(request_dataset)
    data.sort_values(by=['BIT DEPTH'], inplace=True)
    #确定特征值和目标值并进行特征尺度缩放
    features = data[args.input_features]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    save_path_scaler_input = os.path.join(save_dir, 'scaler_input.pkl')
    with open(save_path_scaler_input, 'wb') as f:
        pickle.dump(scaler, f)
    target_scaler = MinMaxScaler()
    target = data[args.output_features]
    target_scaled = target_scaler.fit_transform(target)
    save_path_scaler_output = os.path.join(save_dir, 'scaler_output.pkl')
    with open(save_path_scaler_output, 'wb') as f:
        pickle.dump(target_scaler, f)
    # one_scaler = MinMaxScaler()
    # one_scaled = one_scaler.fit_transform(data[['ROP']])
    # with open('scaler_rop.pkl', 'wb') as f:
    #     pickle.dump(one_scaler, f)
    #建立变量和列索引的字典
    output_colnames = args.output_features
    param = list(output_colnames)
    param_index = range(len(param))
    param_index_dict = dict(zip(param_index, param))
    #进行数据集的划分

    x_train, y_train, x_test, y_test = split_data2(features_scaled, target_scaled, args.window_size)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    # print(x_test)
    # x_train.shape = (186, 19, 5)
    # y_train.shape = (186, 4)
    # x_test.shape = (46, 19, 5)
    # y_test.shape = (46, 4)


    # 注意：pytorch的nn.LSTM input shape=(seq_length, batch_size, input_size)
    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(args.device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(args.device)
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(args.device)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).to(args.device)

    if request_model == 'LSTM':
        model = LSTM(input_dim=len(args.input_features), hidden_dim=args.hidden_dim, num_layers=args.n_layers,
                     output_dim=len(args.output_features))
    elif request_model == 'LSTMAttention_ekan':
        model = LSTMAttention_ekan(input_dim=len(args.input_features), hidden_dim=args.hidden_dim,
                                   output_dim=len(args.output_features), n_layers=args.n_layers, dropout=args.dropout)
    else:
        return None
    #进行训练
    model.to(args.device)
    ## 损失设定
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=request_lr)
    # 使用 TensorDataset 封装训练数据
    train_dataset = TensorDataset(x_train, y_train)

    # 设置 batch_size
    batch_size = 128

    # 使用 DataLoader 来加载训练数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_predict_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    ## 统计MSE均方误差和R2决定系数
    MSE_hist = np.zeros(request_epoch)
    R2_hist = np.zeros(request_epoch)

    ##开始时间统计和结果保存
    start_time = time.time()
    result = []

    ##循环训练
    for t in range(request_epoch):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            # 前向传播
            y_train_pred_batch = model(x_batch)
            # 计算损失
            loss = criterion(y_train_pred_batch, y_batch)
            epoch_loss += loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        average_loss = epoch_loss / len(train_loader)
        print("Epoch ", t, "MSE: ", average_loss)
        # 统计每个时间步的损失和R2指标
        MSE_hist[t] = average_loss

    y_train_pred = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in train_predict_loader:
            x_batch = x_batch.to(args.device)
            y_train_pred_batch = model(x_batch)
            y_train_pred.append(y_train_pred_batch.cpu().numpy())
    training_time = time.time() - start_time
    # 获取当前时间
    # 获取当前时间


    # 使用f-string来格式化文件名
    filename = "model.pth"
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print("Training time: {}".format(training_time))

    # #绘制结果实验图
    # ## 反缩放-恢复正常值
    y_train_pred = np.concatenate(y_train_pred, axis=0)
    predict = pd.DataFrame(target_scaler.inverse_transform(y_train_pred))
    original = pd.DataFrame(target_scaler.inverse_transform(y_train.detach().cpu().numpy()))
    # start_index = 15000
    # print(start_index)
    # # 从该索引开始取500条数据
    # predict_last_200 = predict.iloc[start_index:start_index + 500]
    # predict_last_200.to_csv('temp1.csv')
    # original_last_200 = original.iloc[start_index:start_index + 500]
    # original_last_200.to_csv('original.csv')
    # ##对每个预测特征进行绘图
    # for i in range(len(args.output_features)):
    #     sns.set_style("white")
    #     fig = plt.figure()
    #     fig.subplots_adjust(hspace=0.2, wspace=0.2)
    #     # 计算当前组的行数和列数
    #
    #     ax = sns.lineplot(x=range(len(original_last_200.index)), y=original_last_200.loc[:, i], label=f"{param_index_dict[i]}",
    #                       color='royalblue')
    #     ax = sns.lineplot(x=range(len(predict_last_200.index)), y=predict_last_200.loc[:, i], label=f"Prediction{request_model} ",
    #                       color='tomato')
    #     # ax = sns.lineplot(x=range(len(original.index)), y=original.loc[:, i], label=f"{param_index_dict[i]}",
    #     #                   color='royalblue')
    #     # ax = sns.lineplot(x=range(len(predict.index)), y=predict.loc[:, i], label=f"Training Prediction{args.model_name} ",
    #     #                   color='tomato')
    #     # print(predict.index)
    #     # print(predict[0])
    #
    #     ax.set_title(f'{param_index_dict[i]}', size=14, fontweight='bold')
    #     ax.set_xlabel("Days", size=14)
    #     ax.set_ylabel("unit", size=14)
    #     ax.set_xticklabels('', size=10)
    #
    #     Fitting_path = f'../{args.model_name}/train'
    #     if not os.path.exists(Fitting_path):
    #         # 使用 os.makedirs() 创建目录
    #         os.makedirs(Fitting_path)
    #         print(f"Directory '{Fitting_path}' was created.")
    #     else:
    #         print(f"Directory '{Fitting_path}' already exists.")
    #
    #     plt.savefig(os.path.join(Fitting_path, f'{param_index_dict[i]}'))
    #     plt.show()
    #
    ##绘制loss曲线
    indication_path = save_dir
    plt.plot()
    ax = sns.lineplot(data=MSE_hist, color='royalblue')
    # print(MSE_hist)
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("Training Loss", size=14, fontweight='bold')
    plt.savefig(os.path.join(indication_path, 'loss'))
    plt.show()
    #
    # # plt.plot()
    # # ax = sns.lineplot(data=R2_hist, color='green')
    # # ax.set_xlabel("Epoch", size=14)
    # # ax.set_ylabel("R2", size=14)
    # # ax.set_title("R2", size=14, fontweight='bold')
    # # fig.set_figheight(6)
    # # fig.set_figwidth(16)
    # # plt.savefig(os.path.join(indication_path, 'R2'))
    # # plt.show()

    #进行模型测试
    y_test_pred = model(x_test)
    rmse_train = mean_squared_error(y_train.detach().cpu().numpy(), y_train_pred, squared=False)
    r2_train = r2_score(y_train.detach().cpu().numpy(), y_train_pred)
    mae_train = mean_absolute_error(y_train.detach().cpu().numpy(), y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train.detach().cpu().numpy(), y_train_pred)
    # print('Train RMSE: %.6f ' % (rmse_train))
    # # print('Train R^2: %.2f' % (r2_train))
    # print('Train MAE: %.6f' % (mae_train))
    # y_test_np = y_test.detach().cpu().numpy().ravel()
    # y_test_pred_np = y_test_pred.detach().cpu().numpy().ravel()
    # 创建一个 DataFrame
    # df = pd.DataFrame({
    #     'y_test': y_test_np
    #     'y_test_pred': y_test_pred_np
    # })
    #
    # # 将 DataFrame 输出到 CSV 文件
    # df.to_csv('y_test_vs_y_test_pred.csv', index=False)
    rmse_test = math.sqrt(mean_squared_error(y_test.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy()))
    r2_test = r2_score(y_test.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())
    mae_test = mean_absolute_error(y_test.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())
    mape_test = mean_absolute_percentage_error(y_test.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())
    # print('Test RMSE: %.6f' % (rmse_test))
    # print('Test R^2: %.2f' % (r2_test))
    # print('Test MAE: %.6f' % (mae_test))
    # print('Test MAPE: %.6f' % (mape_test))
    # result.append(rmse_train)
    # result.append(rmse_test)
    # result.append(training_time)

    ## 反缩放-恢复正常值
    # y_train_pred = target_scaler.inverse_transform(y_train_pred)
    # y_train = target_scaler.inverse_transform(y_train.detach().cpu().numpy())
    # y_test_pred = target_scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
    # y_test = target_scaler.inverse_transform(y_test.detach().cpu().numpy())
    # # print("Test RMSE: %.6f" % math.sqrt(mean_squared_error(y_test, y_test_pred)))
    # # print("Test R^2: %.6f" % r2_score(y_test, y_test_pred))
    # # print("Test MAE: %.6f" % mean_absolute_error(y_test, y_test_pred))
    # # print("Test MAPE: %.6f" % mean_absolute_percentage_error(y_test, y_test_pred))
    # #将模型指标添加到model_scores的csv文件中
    # file_path = 'model_scores_new.csv'

    # 创建一个包含当前模型结果的DataFrame
    data_response = {
        "Model": str(request_model),  # 确保为字符串
        "Test RMSE": float(rmse_test),  # 转换为 Python float
        "Test MAE": float(mae_test),  # 转换为 Python float
        "training_time": float(training_time),  # 转换为 Python float
        "Test R^2": float(r2_train),  # 转换为 Python float
    }
    return data_response
    # df = pd.DataFrame(data_response)
    #
    # # 检查文件是否存在
    # if Path(file_path).exists():
    #     # 文件存在，追加数据
    #     df.to_csv(file_path, mode='a', header=False,index=False)
    # else:
    #     # 文件不存在，创建文件并写入数据
    #     df.to_csv(file_path, index=False)
    # #绘制测试结果图
    # for i in range(len(args.output_features)):
    #     trainPredictPlot = np.empty_like(target_scaled[:, i]).reshape(-1, 1)
    #     trainPredictPlot[:, :] = np.nan
    #     trainPredictPlot[args.window_size:len(y_train_pred) + args.window_size, :] = y_train_pred[:, i].reshape(-1, 1)
    #
    #     testPredictPlot = np.empty_like(target_scaled[:, i]).reshape(-1, 1)
    #     testPredictPlot[:, :] = np.nan
    #     testPredictPlot[len(y_train_pred) + args.window_size - 1:len(target_scaled[:, i]) - 1, :] = y_test_pred[:,i].reshape(-1, 1)
    #     original = one_scaler.inverse_transform(target_scaled[:, i].reshape(-1, 1))
    #     # 绘制图形
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(trainPredictPlot, label='Train Prediction', linestyle='-', marker='', color='blue')
    #     plt.plot(testPredictPlot, label='Test Prediction', linestyle='-', marker='', color='green')
    #     plt.plot(original, label='Actual Values', linestyle='-', marker='', color='red')
    #
    #     plt.title(f'Results ({param_index_dict[i]})')
    #     plt.xlabel('Index')
    #     plt.ylabel('Value')
    #     plt.legend()
    #     plt.grid(True)
    #     output_dir = f'../{args.model_name}/test'
    #     if not os.path.exists(output_dir):
    #         # 使用 os.makedirs() 创建目录
    #         os.makedirs(output_dir)
    #         print(f"Directory '{output_dir}' was created.")
    #     else:
    #         print(f"Directory '{output_dir}' already exists.")
    #     # 保存图像
    #     plt.savefig(os.path.join(output_dir, f'{param_index_dict[i]}.png'))
    #     plt.close()
    #
    #     print(f"Plot for {param_index_dict[i]} saved in {os.path.join(output_dir, f'{param_index_dict[i]}.png')}")















    # 获取测试集的开始和结束索引
    # test_start_idx = len(y_train_pred) + args.window_size - 1
    # test_end_idx = len(target_scaled[:, i]) - 1
    #
    # # 对齐 original 中的测试部分
    # original_test_part = original[test_start_idx:test_end_idx]
    #
    # # 确保 testPredictPlot 和 original_test_part 的长度相同
    # assert len(testPredictPlot[test_start_idx:test_end_idx]) == len(original_test_part)
    #
    # # 计算中间 2000 条数据的起始和结束索引
    # middle_start_idx = 3000
    # middle_end_idx = test_end_idx
    # window_length = 9111  # 必须是奇数
    # polyorder = 2
    # # 提取中间部分的 2000 条数据
    # testPredictPlot_middle = testPredictPlot[test_start_idx:test_end_idx][middle_start_idx:middle_end_idx]
    # original_test_middle = original_test_part[middle_start_idx:middle_end_idx]
    # testPredictPlot_middle_smooth = savgol_filter(testPredictPlot_middle, window_length, polyorder, mode='nearest')
    # original_test_middle_smooth = original_test_middle
    # # 绘制中间 2000 条数据的 Test Prediction 图形
    # plt.figure(figsize=(10, 6))
    #
    # # 绘制提取出的中间部分
    # plt.plot(testPredictPlot_middle_smooth, label='Test Prediction', linestyle='-', marker='', color='green')
    # plt.plot(original_test_middle_smooth, label='Actual Values', linestyle='-', marker='', color='red')
    #
    # # 添加标题和标签
    # plt.title(f'Test Results (Middle 2000: {param_index_dict[i]})')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.grid(True)
    # plt.show()



