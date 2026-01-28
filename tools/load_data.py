import os
import numpy as np
import pickle 
import sys 
sys.path.append('../')
from simulator.envs import *

def load_envs_DiDi121(driver_num=2000):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    primary_path = os.path.join(data_dir, "DiDi", "DiDi_day1_grid121.pkl")
    fallback_path = os.path.join(data_dir, "DiDi_day1_grid121.pkl")
    data_path = primary_path if os.path.exists(primary_path) else fallback_path
    with open(data_path, 'rb') as handle:
        data_param = pickle.load(handle) 
    dura_param = data_param['duration'] # 每
    price_param = data_param['price']   # 每级邻居的价格    第0级表示自己网格 0~l_max
    neighbor = data_param['neighbor']    # neighbor>=100 表示不可达的订单
    order_param = data_param['order']   # shape=(11,11,144)  表示 (出发地，目的地，出发时间)
    l_max = 6   # 最大通勤跨邻居数
    M,N = 11,11
    price_param[:,1]/=2     # 减小订单的方差
    np.random.seed(0)
    # 减小order param 数量
    commute1 = order_param.astype(np.float32)
    commute1[(neighbor==0)] = 0
    index = commute1>=3
    commute1[index] = (commute1[index]-3)*0.2+3
    index = commute1>=2
    commute1 = np.round(commute1/2+0.1)
    commute1[neighbor==100]=0
    random_delete = np.random.randint(0,3,(commute1.shape))
    random_delete[commute1.sum(-1).sum(-1)<3000] = 0
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 添加与邻居相关的随机数据
    random_grid_num = np.random.randint(1,6,M*N)
    random_prob = np.zeros((M*N,M*N))
    prob_list = [0.05,0.2,0.4,0.15,0.1,0.05,0.05]
    for k in range(7):
        index= neighbor==k
        random_prob+= prob_list[k]/np.sum(index,axis=-1,keepdims=True)*index
    random_add = np.zeros(commute1.shape)
    for i in range(M*N):
        sample = np.random.choice(M*N,size = (random_grid_num[i],commute1.shape[-1]),replace=True, p=random_prob[i])    
        for t in range(commute1.shape[-1]):
            random_add[i,sample[:,t],t] = 1
    commute1+=random_add
    # 删除数量多的
    random_delete = np.random.randint(0,2,(commute1.shape))
    #random_delete[random_delete<=1] = 0
    index = commute1.sum(1)<=25
    random_delete = random_delete.swapaxes(1,2)
    random_delete[index] = 0
    random_delete = random_delete.swapaxes(1,2)
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 全部删除一点点
    random_delete = np.random.randint(0,5,(commute1.shape))
    random_delete[random_delete<=3] = 0
    random_delete[random_delete>3] = 1
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    order_param = commute1.astype(np.int32)
    # 初始化司机数量
    driver_param=np.zeros(M*N,dtype=np.int32)+1
    order_num = np.sum(np.sum(order_param, axis=1), axis=1)
    driver_param[order_num>=100]= driver_param[order_num>=100]+3
    driver_param[order_num>=400]= driver_param[order_num>=400]+3
    driver_param[order_num>=800]= driver_param[order_num>=800]+3
    driver_param = driver_param*driver_num/np.sum(driver_param)
    driver_param = driver_param.astype(np.int32)
    random_add = np.random.choice(M*N, driver_num-np.sum(driver_param), replace = True)
    for dri in random_add:
        driver_param[dri] += 1
    # 统计数量特别多的网格id
    large_grid_dist={i:0 for i in range(121)}
    a= np.sum(order_param, axis=1)
    for i in range(144):
        b = np.where(a[:,i]>=100)[0]
        for n in b:
            large_grid_dist[n]+=1
    large_grid=[]
    for k,v in large_grid_dist.items():
        if v>0:
            large_grid.append(k)
    print('订单数量: {} , 司机数量: {}'.format( np.sum(order_num), np.sum(driver_param)))

    # 处理为envs的参数
    mapped_matrix_int = np.arange(M*N)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(M,N))
    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(144):
        order_num_dict.append( {i:[order_num[i,t]] for i in range(M*N)} )
    idle_driver_location_mat = np.zeros((144, M*N))
    for t in range(144):
        idle_driver_location_mat[t] = driver_param
    order_time = [0.2, 0.2, 0.15,       # 没用
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]
    n_side = 6
    order_real = []
    onoff_driver_location_mat=[]
    env = CityReal(mapped_matrix_int, order_num_dict, [], idle_driver_location_mat,
                   order_time, price_param, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), neighbor_dis = neighbor , order_param=order_param ,fleet_help=False)
    return env, M, N, None, M*N



def load_envs_NYU143(driver_num=2000):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    primary_path = os.path.join(data_dir, "NYU", "NYU_grid143.pkl")
    fallback_path = os.path.join(data_dir, "NYU_grid143.pkl")
    data_path = primary_path if os.path.exists(primary_path) else fallback_path
    with open(data_path, 'rb') as handle:
        data_param = pickle.load(handle) 
    price_param = data_param['price']   # 每级邻居的价格    第0级表示自己网格 0~l_max
    neighbor = data_param['neighbor']    # neighbor>=100 表示不可达的订单
    order_param = data_param['order']   # shape=(11,11,144)  表示 (出发地，目的地，出发时间)
    M,N = data_param['shape']
    l_max = 6   # 最大通勤跨邻居数
    # 减小order param 数量
    np.random.seed(0)
    commute1 = order_param.astype(np.float32)
    commute1[(neighbor==0)] = 0
    index = commute1>=3
    commute1[index] = (commute1[index]-3)*0.2+3
    index = commute1>=2
    commute1 = np.round(commute1/2+0.1)
    commute1[neighbor==100]=0
    random_delete = np.random.randint(0,3,(commute1.shape))
    random_delete[commute1.sum(-1).sum(-1)<3000] = 0
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    random_delete = np.random.randint(0,2,(commute1.shape[1],commute1.shape[2]))
    commute1[68]-=random_delete
    commute1[commute1<0] = 0
    # 添加与邻居相关的随机数据
    random_grid_num = np.random.randint(1,6,M*N)
    random_prob = np.zeros((M*N,M*N))
    prob_list = [0.05,0.2,0.4,0.15,0.1,0.05,0.05]
    for k in range(7):
        index= neighbor==k
        random_prob+= prob_list[k]/np.sum(index,axis=-1,keepdims=True)*index
    random_add = np.zeros(commute1.shape)
    for i in range(M*N):
        sample = np.random.choice(M*N,size = (random_grid_num[i],commute1.shape[-1]),replace=True, p=random_prob[i])    
        for t in range(commute1.shape[-1]):
            random_add[i,sample[:,t],t] = 1
    commute1+=random_add
    # 删除数量多的
    random_delete = np.random.randint(0,2,(commute1.shape))
    #random_delete[random_delete<=1] = 0
    index = commute1.sum(1)<=25
    random_delete = random_delete.swapaxes(1,2)
    random_delete[index] = 0
    random_delete = random_delete.swapaxes(1,2)
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 全部删除一点点
    random_delete = np.random.randint(0,5,(commute1.shape))
    random_delete[random_delete<=3] = 0
    random_delete[random_delete>3] = 1
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    order_param = commute1.astype(np.int32)
    # 初始化司机数量
    driver_param=np.ones(M*N,dtype=np.int32)
    driver_param = driver_param*driver_num/np.sum(driver_param)
    driver_param = driver_param.astype(np.int32)
    random_add = np.random.choice(M*N, driver_num-np.sum(driver_param), replace = True)
    for dri in random_add:
        driver_param[dri] += 1
    # 统计数量特别多的网格id
    large_grid_dist={i:0 for i in range(121)}
    a= np.sum(order_param, axis=1)
    for i in range(144):
        b = np.where(a[:,i]>=100)[0]
        for n in b:
            large_grid_dist[n]+=1
    large_grid=[]
    for k,v in large_grid_dist.items():
        if v>0:
            large_grid.append(k)
    order_num = order_param.sum()
    print('订单数量: {} , 司机数量: {}'.format( np.sum(order_num), np.sum(driver_param)))

    # 处理为envs的参数
    mapped_matrix_int = np.arange(M*N)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(M,N))
    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(144):
        order_num_dict.append( {i:[order_num[i,t]] for i in range(M*N)} )
    idle_driver_location_mat = np.zeros((144, M*N))
    for t in range(144):
        idle_driver_location_mat[t] = driver_param
    order_time = [0.2, 0.2, 0.15,       # 没用
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]
    n_side = 6
    order_real = []
    onoff_driver_location_mat=[]
    env = CityReal(mapped_matrix_int, order_num_dict, [], idle_driver_location_mat,
                   order_time, price_param, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), neighbor_dis = neighbor , order_param=order_param ,fleet_help=False)
    return env, M, N, None, M*N


def load_envs_custom(data_path, driver_num=2000, use_real_orders=False, real_order_sample_rate=0.1):
    """
    加载自定义数据集
    
    Args:
        data_path: PKL 数据文件路径
        driver_num: 司机数量
        use_real_orders: 是否使用真实订单数据（包含真实价格）
        real_order_sample_rate: 真实订单采样率 (0.0-1.0)，默认 0.1 (10%)
    """
    with open(data_path, 'rb') as handle:
        data_param = pickle.load(handle)
    neighbor = data_param['neighbor']
    price_param = data_param['price']
    order_param = data_param['order']
    shape = data_param.get('shape', None)
    
    # 加载真实订单数据（如果有且启用）
    real_orders_data = data_param.get('real_orders', None)
    if use_real_orders and real_orders_data is not None and len(real_orders_data) > 0:
        order_real_full = real_orders_data.tolist() if hasattr(real_orders_data, 'tolist') else list(real_orders_data)
        
        # 对真实订单进行采样以提高训练速度
        if real_order_sample_rate < 1.0:
            np.random.seed(42)  # 固定种子保证可复现
            sample_size = max(1, int(len(order_real_full) * real_order_sample_rate))
            indices = np.random.choice(len(order_real_full), sample_size, replace=False)
            order_real = [order_real_full[i] for i in indices]
            print(f"[load_envs_custom] 真实订单采样: {len(order_real_full)} -> {len(order_real)} ({real_order_sample_rate*100:.0f}%)")
        else:
            order_real = order_real_full
            print(f"[load_envs_custom] 使用全部真实订单: {len(order_real)} 条")
        
        if len(order_real) > 0:
            prices = [o[4] for o in order_real]
            print(f"  真实价格统计: mean={np.mean(prices):.2f}, std={np.std(prices):.2f}")
    else:
        order_real = []
        if use_real_orders:
            print(f"[load_envs_custom] 警告: 数据文件中没有 real_orders，将使用采样价格")

    grid_num = int(order_param.shape[0])
    if shape is None:
        M, N = grid_num, 1
    else:
        M, N = shape
        if M * N != grid_num:
            M, N = grid_num, 1

    l_max = int(np.max(neighbor[neighbor < 100])) if np.any(neighbor < 100) else 1
    l_max = max(1, min(l_max, 8))

    order_param = order_param.astype(np.int32)
    total_orders = int(order_param.sum())
    print(f"[load_envs_custom] order_param total_orders: {total_orders}")

    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(order_param.shape[2]):
        order_num_dict.append({i: [order_num[i, t]] for i in range(grid_num)})

    origin_volume = order_param.sum(axis=(1, 2)).astype(np.float64)
    if origin_volume.sum() > 0:
        driver_param = origin_volume / origin_volume.sum() * driver_num
    else:
        driver_param = np.ones(grid_num, dtype=np.float64) * driver_num / grid_num
    driver_param = np.floor(driver_param).astype(np.int32)
    remainder = driver_num - int(driver_param.sum())
    if remainder > 0:
        add_idx = np.random.choice(grid_num, remainder, replace=True)
        for idx in add_idx:
            driver_param[idx] += 1

    idle_driver_location_mat = np.zeros((order_param.shape[2], grid_num))
    for t in range(order_param.shape[2]):
        idle_driver_location_mat[t] = driver_param

    order_time = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.04, 0.01]
    n_side = 6
    onoff_driver_location_mat = []
    mapped_matrix_int = np.arange(M * N).reshape(M, N)
    
    # 设置采样概率（如果使用真实订单，概率为 1 表示使用全部订单）
    probability = 1.0 if use_real_orders and len(order_real) > 0 else 1.0 / 28
    prob_env = os.environ.get("COOPRIDE_ORDER_PROB")
    if prob_env:
        try:
            probability = float(prob_env)
            print(f"[load_envs_custom] override probability={probability}")
        except ValueError:
            print(f"[load_envs_custom] invalid COOPRIDE_ORDER_PROB={prob_env}, keep {probability}")
    
    env = CityReal(
        mapped_matrix_int,
        order_num_dict,
        [],
        idle_driver_location_mat,
        order_time,
        price_param,
        l_max,
        M,
        N,
        n_side,
        order_param.shape[2],
        probability,
        np.array(order_real),
        np.array(onoff_driver_location_mat),
        neighbor_dis=neighbor,
        order_param=order_param,
        fleet_help=False,
    )
    
    # 如果使用真实订单，预先生成 day_orders
    if use_real_orders and len(order_real) > 0:
        env.utility_bootstrap_oneday_order()
        print(f"[load_envs_custom] 已预生成 day_orders")
    
    return env, M, N, None, grid_num



if __name__ == '__main__':     

    load_envs_DiDi121(2000)
    #load_envs_NYU143(2000)
