import numpy as np
from sklearn.cluster import DBSCAN

def calculate_distance(traj1, traj2):
    """
    计算两个轨迹之间的距离，这里使用欧氏距离作为度量
    """
    return np.linalg.norm(traj1 - traj2)

def dbscan_clustering(trajectories, eps, min_samples):
    """
    DBSCAN 聚类算法实现

    参数：
    trajectories: 轨迹数据，一个二维数组，每行表示一个轨迹
    eps: ϵ-邻域的半径，DBSCAN 参数
    min_samples: 最小点数，DBSCAN 参数
    
    返回：
    clusters: 每个轨迹的聚类标签，-1 表示噪声点
    """
    # 计算轨迹之间的距离矩阵
    num_trajectories = len(trajectories)
    distances = np.zeros((num_trajectories, num_trajectories))
    for i in range(num_trajectories):
        for j in range(i+1, num_trajectories):
            distances[i, j] = calculate_distance(trajectories[i], trajectories[j])
            distances[j, i] = distances[i, j]  # 对称性

    # 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # 使用预计算的距离矩阵
    clusters = dbscan.fit_predict(distances)
    
    return clusters

a = np.array([[0,1,2],[1,2,3],[2,2,3],[3,1,5],[4,2,5]])
b = np.array([[0,3,5],[1,2,4],[2,5,4],[3,1,2],[4,3,4]])
c = np.array([[0,1,4],[1,2,3],[2,2,5],[3,1,5],[4,3,5]])
d = np.array([[0,3,5],[1,2,4],[2,5,4],[3,4,2],[4,2,4]])
e = np.array([[0,2,4],[1,1,3],[2,2,2],[3,1,4],[4,2,5]])
f = np.array([[0,1,4],[1,2,2],[2,3,4],[3,3,2],[4,1,3]])
g = np.array([[0,1,2],[1,2,3],[2,2,3],[3,1,5],[4,2,5]])
h = np.array([[0,3,5],[1,2,4],[2,5,4],[3,1,2],[4,3,4]])
i = np.array([[0,1,4],[1,2,3],[2,2,5],[3,1,5],[4,3,5]])
j = np.array([[0,3,5],[1,2,4],[2,5,4],[3,4,2],[4,2,4]])
k = np.array([[0,2,4],[1,1,3],[2,2,2],[3,1,4],[4,2,5]])
l = np.array([[0,1,4],[1,2,2],[2,3,4],[3,3,2],[4,1,3]])
user_data = {'user1':{'video1': a,'video2':b},'user2':{'video1':c,'video2':d},
             'user3':{'video1':e,'video2':f},'user4':{'video1':g,'video2':h},
             'user5':{'video1':i,'video2':j},'user6':{'video1':k,'video2':l}}

'''用户数据的调用'''
def User_clustering(original_dataset):
    dataset = {}
    xun_lian_list = [] 
    data_per_video = []        
    for enum_user,user in enumerate(original_dataset.keys()):        
            dataset[user] = {}
            for enum_video,video in enumerate(original_dataset[user].keys()):              
                  
                data_per_video.append(original_dataset[user][video])
                xun_lian_list = np.array(data_per_video) 
                break         
    return xun_lian_list  


'''聚类和结果'''

trajectories = User_clustering(user_data)
print("轨迹数据：")
print(trajectories)

# DBSCAN 聚类
eps = 5  # 邻域半径
min_samples = 2  # 最小点数
clusters = dbscan_clustering(trajectories, eps, min_samples)

# 打印聚类结果
print("\n轨迹聚类结果：")
for idx, traj in enumerate(trajectories):
    print(f"轨迹{idx+1}的聚类标签：{clusters[idx]}")
