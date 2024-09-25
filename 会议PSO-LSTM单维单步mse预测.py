import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#用前seq_len天的4个特征预测后mul天的adjclose这一项特征
#gpu,batch_size不能太大或者太小,1个epoch相当于把全部训练数据集训练了一遍,batch_size相当于一口气学几行(天).
# 学完一个batch更新一次神经网络权重,1个epoch学(总行数/batch_size)次,更新这么多次.多个epoch相当于一本书复习多遍,记得就牢了,融会贯通了

#para,可修改!!!!!!!!!!!!!!
seq_len = 20  # Sequence_length(10)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mulpre = 1 #多步预测预测的天数(3)!!!!!!!!!!!!!!!!!!!!!!!!!!
pN = 3  #规定初始种群个数 (8)!!!!!!!!!!!!!!!!!!!!!!!!!!!!
iters = 3#规定IPSO迭代次数(10)!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
batch_size = 64#规定batch大小(64)
division_rate1 = 0.9#分割训练集占比,剩下的是测试集
division_rate2 = 1.0

#formula,可修改!!!!!!!!!!!!!!!!!PSO中的w1,w2,c1,c2三个公式分开
formula_w1 = '0.5 + 1/(1+math.exp(10*t/iters))'
formula_w2 = '0.5 + 1/(1+math.exp(10*t/iters))'#公式,每次更改需要修改!!!!!!!!!!!!!!!!!!!!!!!
formula_c1 = 'c1_start+(c1_end-c1_start)*np.sqrt(np.tanh(np.pi*t/iters))'
formula_c2 = 'c2_start+(c2_end-c2_start)*np.sqrt(np.tanh(np.pi*t/iters))'

l = ['000001.SS', 'AAPL', 'BTC-USD' , 'DJI', 'Gold_daily','GSPC','IXIC']

for i in l:
    #path
    file_path = 'C:/lyx/learning/期刊论文/程序结果/Informer/' + i#要保存图片和excel的那个文件夹,每次运行需要修改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #PSO中公式参数
    c1_start = 2.55
    c1_end = 1.4
    c2_start = 1.5
    c2_end = 2.45
    r1 = 0.8
    r2 = 0.3
    # 公式里的

    if(os.path.exists(file_path)):
        print('文件夹已存在')
    else:
        os.makedirs(file_path)

    picture_path = file_path + '/prediction_result.svg'#预测拟合图像保存路径,最后一项不用改
    picture_path2 = file_path + '/attention_result'#注意力权重图像保存路径,最后一项不用改
    picture_path3 = file_path + '/loss_result'#损失图像保存路径,最后一项不用改
    excel_path = file_path +'/accu.xls'#最终的准确率和各项指标存放的路径和名称
    para_path = file_path + '/para.xls'#训练的神经网络参数
    stock_name = i
    sdate = '2010-11-01'
    edate = '2020-10-31'
    d = 0.2  # Dropout
    shape = [1, seq_len, 1]  # feature, window, output,和loaddata划分数据集还有buildmodeal有大关系!!!
    # neurons = [?, ?, mulpre]
    epochs = 100
    filename = 'C:/lyx/learning/会议论文/三支同时期数据/' + i + '.csv'
    final = []#最终的最佳特征数组
    tlist = []#迭代次数序列
    tresult = {}#每次迭代的最小fitness
    accuracy = 0


    #gpu设置
    '''os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # 使用编号为1，2号的GPU
    config = tf.compat.v1.ConfigProto
    
    config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
    session = tf.Session(config=config)
    
    # 设置session
    KTF.set_session(session )'''

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    #获取股票数据dataframe
    def get_stock_data(normalize=True):
        df = pd.read_csv(filename, usecols=['Adj Close'])
        if normalize:  # 归一化
            standard_scaler = preprocessing.StandardScaler()
            df['Adj Close'] = standard_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
        return df

    #先划分训练集测试集,再标准化归一化,避免数据泄露
    def load_data(df, seq_len , mul, normalize=True):
        amount_of_features = len(df.columns)  # columns是列索引,index是行索引
        data = df.values
        row1 = round(division_rate1 * data.shape[0])  # 70% split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
        row2 = round(division_rate2 * data.shape[0])
        #训练集和测试集划分
        train = data[:int(row1), :]
        test = data[int(row1):int(row2), :]

        # 训练集和测试集归一化
        if normalize:
            standard_scaler = preprocessing.StandardScaler()
            train = standard_scaler.fit_transform(train.reshape(-1, 1))
            test = standard_scaler.fit_transform(test.reshape(-1, 1))

        X_train = []  # train列表中4个特征记录
        y_train = []
        X_test = []
        y_test = []
        train_samples=train.shape[0]-seq_len-mul+1
        test_samples = test.shape[0] - seq_len - mul + 1
        for i in range(0,train_samples,mul):  # maximum date = lastest date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
            X_train.append(train[i:i + seq_len,])#每个滑窗每天四个特征
            y_train.append(train[i+seq_len:i+seq_len+mul,-1])#-1即取最后一个特征

        for i in range(0, test_samples,mul):  # maximum date = lastest date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
            X_test.append(test[i:i + seq_len, ])  # 每个滑窗每天四个特征
            y_test.append(test[i + seq_len:i + seq_len + mul, -1])  # -1即取最后一个特征
        # X都对应全部4特征,y都对应adj close   #train都是前百分之90,test都是后百分之10
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print('train', train.shape)
        print(train)
        print('test', test.shape)
        print(test)
        print('X_train', X_train.shape)
        print('y_train', y_train.shape)
        print('X_test', X_test.shape)
        print('y_test', y_test.shape)
        print('df', df)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))  # (90%maximum, seq-1 ,4) #array才能reshape
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # (10%maximum, seq-1 ,4) #array才能reshape
        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        return [X_train, y_train, X_test, y_test],row1,row2  # x是训练的数据，y是数据对应的标签,也就是说y是要预测的那一个特征!!!!!!


    #普通可视化
    # Draw Plot
    def plot_df(df, title="", xlabel='Date', ylabel='Adj Close', dpi=100):
        plt.ion()
        plt.figure(figsize=(16, 5), dpi=dpi)
        x = df.index
        y = df['Adj Close']
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()
        plt.close()

    # Time series data source: fpp pacakge in R.
    #df = pd.read_csv(filename, parse_dates=['Date'],usecols=['Date','Open','High','Low','Adj Close', 'Volume'])

    '''head = df.head()
    print('head\n',head)
    plot_df(df, title='Adj Close of SP500ETF from 2010 to 2020.')'''



    def build_model(shape, neurons, d, decay):
        model = Sequential()

        model.add(LSTM(neurons[0], input_shape=(shape[1], shape[0]), return_sequences=True))
        model.add(Dropout(d))

        model.add(LSTM(neurons[1], return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(neurons[2]))

        adam = optimizers.Adam(decay=decay)
        # lr = lr\(1 + decay * iterations),adam自动逐渐减小学习率
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])  # loss函数可调节,可以增加准确率
        model.summary()
        return model


    def measure(d, shape, neurons, epochs, decay):
        tf.keras.backend.clear_session()#清理模块!!!!!!!!!!!!!!
        model = build_model(shape, neurons, d, decay)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
        trainScore = model.evaluate(X_train, y_train, verbose=0)  # 不输出日志信息
        print('Loss: %.5f MSE ' % trainScore[0])
        print('Accuracy: %.3f MSE ' % trainScore[1])
        return trainScore[0]


    def fitness(X, i):
        epochs = int(X[i][0])
        decay = X[i][1]
        neurons = [int(X[i][2]), int(X[i][3]), mulpre]
        score = measure(d, shape, neurons, epochs, decay)
        score = np.sqrt(score)#这里用rmse而不是mse因为归一化后mse太小收敛太快
        return score

    '''def calculate_MAPE(pre,real):
        sum = 0
        for u in range(len(real)):  # for each data index in test data
            pr = pre[u][0]  # pr = prediction on day u
            re = real[u][0]
            sum += (abs(pr - re) / re) / re
        MAPE = (sum/len(real))*100
        return MAPE
    '''
    #计算涨跌趋势准确率
    def single_up_down_accuracy(pre, real):
        real_var = real[1:] - real[:len(real) - 1]  # 实际涨跌
        pre_var = pre[1:] - pre[:len(pre) - 1]  # 原始涨跌
        txt = np.zeros(len(real_var))
        for i in range(len(real_var - 1)):  # 计算数量
            txt[i] = (np.sign(real_var[i]) == np.sign(pre_var[i]))
        result = sum(txt) / len(txt)
        return result

    #计算multi天内涨跌趋势准确率
    def multi_up_down_accuracy(pre, real):
        real_var = real[mulpre:] - real[:len(real) - mulpre]  # 实际涨跌
        pre_var = pre[mulpre:] - pre[:len(pre) - mulpre]  # 原始涨跌
        txt = np.zeros(len(real_var))
        for i in range(len(real_var - 1)):  # 计算数量
            txt[i] = (np.sign(real_var[i]) == np.sign(pre_var[i]))
        result = sum(txt) / len(txt)
        return result

    #计算最终的准确率和各项指标,并且保存到文件夹excel中,(1-每日的相对误差率)计算算数平均值
    def calculate_accuracy(pre, real):
        accuracy = 0
        for u in range(len(real)):  # for each data index in test data
            pr = pre[u][0]  # pr = prediction on day u
            accuracy += ((1 - abs(pr - real[u]) / real[u])) / len(real)
            #percentage_diff.append((pr - y_train[u] / pr) * 100)

        single_trend_accuracy = single_up_down_accuracy(pre,real)
        multi_trend_accuracy = multi_up_down_accuracy(pre, real)

        # MAPE = np.mean(np.abs((pre - real) / real))
        MAPE = sklearn.metrics.mean_absolute_percentage_error(real,pre)
        #MAPE = calculate_MAPE(pre,real)
        RMSE = np.sqrt(np.mean(np.square(pre - real)))
        MAE = np.mean(np.abs(pre - real))
        R2 = r2_score(pre, real)
        dict = {'single_trend_accuracy': single_trend_accuracy, 'multi_trend_accuracy': multi_trend_accuracy, 'accuracy': accuracy, 'MAPE': MAPE, 'RMSE': RMSE, 'MAE': MAE, 'R2': R2}
        df = pd.DataFrame(dict)
        print('最终的准确率和指标如下\n',df)
        return df



    # 反归一化,这里有问题,用训练集和测试集一起反归一化了!!!!!!!!!!!!!!!!!!!!!!!
    def denormalize(normalized_value):
        df = pd.read_csv(filename)
        data = df['Adj Close'].values.reshape(-1, 1)
        row1 = round(division_rate1 * data.shape[0])  # 70% split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
        row2 = round(division_rate2 * data.shape[0])
        # 训练集和测试集划分
        test = data[int(row1):int(row2), :]
        normalized_value = normalized_value.reshape(-1, 1)

        standard_scaler = preprocessing.StandardScaler()
        '反归一化'
        std = standard_scaler.fit_transform(test)  # 利用m对data进行归一化，并储存df的归一化参数
        new = standard_scaler.inverse_transform(normalized_value)  # 利用m对normalized_value进行反归一化

        '归一化'
        '''m = min_max_scaler.fit_transform(train)  # 利用m对train进行归一化，并储存df的归一化参数!!
        new = min_max_scaler.transform(test)  # 利用m对test也进行归一化,注意这里是transform不能是fit_transform!!!1'''
        return new


    # 可视化  'plt原来是plt2'
    def plot_result(pre, real):
        real100 = real[0:101]
        pre100 = pre[0:101]
        plt.figure(figsize=(6.4, 4.8), dpi=2000)
        plt.plot(pre100, color='red', label='Prediction', linewidth=0.6)
        plt.plot(real100, color='blue', label='Actual', linewidth=0.6)
        plt.rcParams.update({'font.size': 20})
        plt.legend(loc='best')
        plt.title('The test result for {}'.format(stock_name), fontsize = 20)
        plt.xlabel('Days', fontsize = 20)
        plt.ylabel('Adjusted Closing Price', fontsize = 20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        result = np.array(pre100)
        result2 = np.array(real100)
        #days = len(result)
        min2 = min(result)
        min3 = min(result2)
        if (min3 <= min2):
            min2 = min3
        max2 = max(result)
        max3 = max(result2)
        if (max3 >= max2):
            max2 = max3
        plt.xlim([0, 101])
        plt.ylim([min2, max2])
        plt.savefig(picture_path, dpi=2000)  # 保存拟合曲线
        plt.clf()


    #plot_result(p, y_test) y_test(83,3)
    #plot_result(p, y_train)


    def boundary(pop):  # 边界判断,pop是一个列表
        # 防止粒子跳出范围
        # 迭代数和节点数都应为整数,pop[1]即学习率不需要是整数
        pop[0] = int(pop[0])  # 迭代次数
        pop[2] = int(pop[2])  # 第一个隐含层节点数
        pop[3] = int(pop[3])  # 第二个隐含层节点数
        if pop[0] > 200 or pop[0] < 1:
            pop[0] = np.random.randint(1, 200)
        if pop[1] > 0.01 or pop[1] < 0.001:
            pop[1] = (0.01 - 0.001) * np.random.rand() + 0.001
        if pop[2] > 200 or pop[2] < 1:
            pop[2] = np.random.randint(1, 200)
        if pop[3] > 200 or pop[3] < 1:
            pop[3] = np.random.randint(1, 200)

        return pop  # 返回修正后的特征列表


    def MYPSO(max_iter):
        # MYPSO参数设置  max_iter = 10
        dim = 4  # 搜索维度,第一个维度是迭代次数[1-200]
        # 第二个是学习率[0.001-0.01]
        # 第三和第四个是隐含层节点数[1-200]
        # 这些参数就不说了

        # 初始化
        X = np.zeros((pN, dim))  # 5*4的全0矩阵,4个维度特征
        V = np.zeros((pN, dim))
        pbest = np.zeros((pN, dim))
        gbest = np.zeros((1, dim))  # 1*4的全0矩阵,唯一全局最大值位置粒子的四个特征,best记录四个特征!
        p_fit = np.zeros(pN)  # 5行1列,记录5个粒子当前的fitness
        result = np.zeros((max_iter, dim))  # 每次迭代gbest那个粒子的4个特征
        fit = 10000  # 随便写一个很大的初始值,以后记录gbest粒子的最小fitness值
        for i in range(pN):
            for j in range(dim):  # 5个粒子的特征全部随机生成
                if j == 0:
                    X[i][j] = np.random.randint(1, 200)  # 产生1-200的整数
                    print(X[i][j])
                elif j == 1:
                    X[i][j] = (0.01 - 0.001) * np.random.rand() + 0.001  # 产生0.001-0.01的小数
                elif j == 2:
                    X[i][j] = np.random.randint(1, 200)  # 产生1-200的整数
                elif j == 3:
                    X[i][j] = np.random.randint(1, 200)

                V[i][j] = np.random.rand()
            pbest[i] = X[i].copy()  # 初始化best是初始位置的4个特征

            tmp = fitness(X, i)#当前值为none不为int

            p_fit[i] = tmp
            if (tmp < fit):
                fit = tmp
                gbest = X[i].copy()  # 当前粒子的fitness最小,则记录特征
                # 到此位置只是确定所有初始值
        # 开始循环迭代
        trace = []  # 踪迹,一个一维列表记录每次迭代中全局最优对应的最小fitness
        for t in range(max_iter):
            tlist.append(t)
            for i in range(pN):  # 更新gbest\pbest
                temp = fitness(X, i)
                if (temp < p_fit[i]):  # 更新个体最优
                    p_fit[i] = temp
                    pbest[i, :] = X[i, :].copy()
                    if (p_fit[i] < fit):  # 更新全局最优,如果不是个体最优就更不用考虑更新全局最优了
                        gbest = X[i, :].copy()
                        fit = p_fit[i].copy()
            #对应开头的w,c1,c2公式!!!
            if(t <= 0.2 * iters):
                w = eval(formula_w1)
            else:
                w = eval(formula_w2)  # 创新点！！！！！w每次迭代变化,所有粒子共享变化
            c1 = eval(formula_c1)
            c2 = eval(formula_c2)
            for i in range(pN):
                V[i, :] = w * V[i, :] + c1 * r1 * (pbest[i] - X[i, :]) + c2 * r2 * (gbest - X[i, :])  # 只有w每次变化
                X[i, :] = X[i, :] + V[i, :]
                X[i, :] = boundary(X[i, :])  # 边界判断
                # 加入自适应变异操作，避免陷入局部最优
                prob = 0.5 * t / max_iter + 0.5  # 自适应变异，随着进化代数的增加，变异几率越小     #创新点！！！！
                if np.random.rand() > prob:  # 满足条件就会变异,增加探索的范围,避免陷入局部最优?prob是probablity概率
                    for j in range(dim):
                        if j == 0:
                            X[i][j] = np.random.randint(1, 200)  # 产生1-200的整数
                        elif j == 1:
                            X[i][j] = (0.01 - 0.001) * np.random.rand() + 0.001  # 产生0.001-0.01的小数
                        elif j == 2:
                            X[i][j] = np.random.randint(1, 200)  # 产生1-200的整数
                        elif j == 3:
                            X[i][j] = np.random.randint(1, 200)
                            # 以上部分自适应变异，随着进化代数的增加，变异几率越小     #创新点！！！！该粒子的所有特征重新随机初始化(变异)
            result[t, :] = gbest.copy()  # 记录这次迭代中全局最佳粒子的所有特征
            trace.append(fit)  # 记录每次迭代中全局最优对应的最小fitness
            tresult[t] = fit
            print('这是第t次迭代后最佳的全部数据:t,fit,gbest')
            print(t, fit, gbest)  # 输出第t次迭代最佳的全部信息
            final = gbest
        return final,tlist,tresult

    #可视化每次迭代的loss曲线
    def plot_loss(loss):

        lists = sorted(loss.items())
        x, y = zip(*lists)#p = [[1,2,3],[4,5,6]]   zip(*p) = [(1, 4), (2, 5), (3, 6)]   zip(p) = [([1, 2, 3],), ([4, 5, 6],)]
        plt.plot(x, y)
        plt.title('The loss as t changes')
        plt.xlabel('t')
        plt.ylabel('loss')
        plt.savefig(picture_path3)
        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()  # 自动关闭
        #plt.figure()
        plt.clf()

    def save_data(accu,model):
        accu.to_excel(excel_path)#保存准确率和指标
        para = {'division_rate1': division_rate1, 'division_rate2': division_rate2,'seq_len': seq_len, 'mulpre': mulpre, 'pN': pN, 'iters': iters,'batch_size' : batch_size}
        para = pd.DataFrame([para])
        para.to_excel(para_path)#保存网络参数
        '''model_path = str(file_path) + '/model.png'
        plot_model(model, to_file= model_path)#保存网络结构'''
        formula_path = str(file_path) + '/formula.txt'
        with open(formula_path, 'w') as f:#保存公式
            f.write('c1_start = ' + str(c1_start) + '\tc1_end = ' + str(c1_end) + '\tc2_start = ' + str(c2_start) + '\tc2_end' + str(c2_end) + '\tr1 = ' + str(r1) + '\tr2' + str(r2) + 'w1 = ' + formula_w1 + '\n' + 'w2 = ' + formula_w2 + '\n' + 'c1 = ' + formula_c1 + '\n' + 'c2 = ' + formula_c2)

    #数据整理
    df = get_stock_data()  # 修改了
    #print('df.shape',df.shape)
    [X_train, y_train, X_test, y_test],row1,row2 = load_data(df, seq_len, mulpre)
    #corr_heatmap(df)暂时不用热力图

    #IPSO得出final即最佳参数集
    final, tlist, tresult = MYPSO(iters)  # 规定几次迭代!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #loss随着迭代次数t变化轨迹可视化,暂时不用
    plot_loss(tresult)
    #数据整理
    final[0] = int(final[0])
    final[2] = int(final[2])
    final[3] = int(final[3])
    print('final', final)
    epochs = int(final[0])
    decay = final[1]
    neurons = [int(final[2]), int(final[3]), mulpre]  # neurons[3]=mul,输出三个值!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #训练并且测试得出的最佳神经网络
    model = build_model(shape, neurons, d, decay)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)#batch_size不能太大或者太小!!!!!!!!!!!!!!!
    trainScore = model.evaluate(X_test, y_test, verbose=0)
    p = model.predict(X_test)
    print('pre.shape',p.shape)
    p = denormalize(p)
    y_test = denormalize(y_test)


    print('pre.shape',p.shape)#(735, 1)
    print('pre',p)
    print('real',y_test)
    print('real.shape',y_test.shape)#(735, 1)

    stock = i
    model2 = 'LSTM'
    csv_path = 'C:/lyx/learning/期刊论文/程序结果/对比图表/' + stock +'/' + model2 + '.xls'
    df = pd.DataFrame(p)
    df.columns.name = None
    df.to_excel(csv_path,index=False,header=None)

    #计算所有指标并且保存
    accu = calculate_accuracy(p,y_test)
    save_data(accu,model)
    #用test测试最终效果, 并且绘制贴近图线
    #plot_result(p, y_test)









