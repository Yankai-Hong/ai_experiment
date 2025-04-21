# 遗传算法计算TSP问题
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import time


class GeneticAlgTSP:
    def __init__(self, file_name_, modified_times_, remain_rate_,
                 select_rate_, count_, mutation_rate_, iteration_times_):

        self.file_name = file_name_
        self.modified_times = modified_times_
        self.remain_rate = remain_rate_
        self.select_rate = select_rate_
        self.count = count_
        self.mutation_rate = mutation_rate_
        self.iteration_times = iteration_times_

        self.childs = []
        self.parents = []
        self.population = []
        self.coord = []
        self.distance = []

    # 读取文件数据 返回coord矩阵
    def cope_data(self):
        # 取后面两个列为坐标值
        self.coord = np.loadtxt(self.file_name, usecols=(1, 2))

    # 获取distance矩阵
    def get_distance(self):
        self.distance = np.linalg.norm(self.coord[:, np.newaxis] - self.coord[np.newaxis, :], axis=2)

    # 适应度函数 --> 计算path环路的距离
    def get_value(self, path):
        # 相邻点求和 + 最后一段闭合路线
        return np.sum(self.distance[path[:-1], path[1:]]) + self.distance[path[-1], path[0]]

    # 修复 modified 函数，避免频繁调用 get_value
    def modified(self, path):
        t = 0
        best_path = path.copy()
        best_value = self.get_value(best_path)  # 计算初始路径的值

        while t < self.modified_times:
            mov = path.copy()
            a, b = random.sample(range(len(path)), 2)  # 使用 random.sample 确保 a != b
            mov[a], mov[b] = mov[b], mov[a]  # 直接交换

            new_value = self.get_value(mov)
            if new_value < best_value:  # 仅在路径优化时更新
                best_path = mov
                best_value = new_value
            t += 1

        return best_path

    # 初始化种群
    def init(self, len_path):
        self.population = []
        path = [i for i in range(len_path)]

        while len(self.population) < self.count:
            tmp = path.copy()
            random.shuffle(tmp)
            tmp = self.modified(tmp)
            self.population.append(tmp)

    # 自然选择
    def natural_select(self):
        # 根据每一个path的value值排序
        sorted_population = [x for x in sorted(self.population, key=lambda x: self.get_value(x))]

        remain_num = int(self.remain_rate * len(self.population))
        self.parents = sorted_population[:remain_num]

        # 剩余中随机挑选
        for n in sorted_population[remain_num:]:
            if random.random() < self.select_rate:
                self.parents.append(n)

    # 交叉繁殖
    def cross(self):
        target = self.count - len(self.parents)
        self.childs = []

        while len(self.childs) <= target:
            male_index, female_index = random.sample(range(len(self.parents)), 2)

            # 随机选取父母基因
            male = self.parents[male_index]
            female = self.parents[female_index]

            if len(male) < 2 or len(female) < 2:
                break

            # 获取截取基因片段的左右坐标
            left = random.randint(0, len(male) - 2)
            right = random.randint(left + 1, len(male))
            # 截取基因片段
            male_gene = male[left:right]
            female_gene = female[left:right]

            # 优化基因交叉逻辑
            child_a = []
            child_a.extend(female_gene)
            for i in male:
                if i not in female_gene:
                    child_a.append(i)

            child_b = []
            child_b.extend(male_gene)
            for i in female:
                if i not in male_gene:
                    child_b.append(i)

            self.childs.append(child_a)
            self.childs.append(child_b)

    # 基因变异
    def mutation(self):
        for child in self.childs:
            if random.random() < self.mutation_rate and len(child) >= 2:
                # 直接选取两个不同索引
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]

    # 返回population的最小值及最短路线
    def get_result(self):
        sorted_population = [x for x in sorted(self.population, key=lambda x: self.get_value(x))]
        return sorted_population[0].copy(), self.get_value(sorted_population[0])

    # 优化 iterate 函数，添加进度输出
    def iterate(self):
        self.cope_data()
        self.get_distance()

        row = self.coord.shape[0]
        self.init(row)

        cur_dis = float('inf')
        cur_path = []

        # 记录开始训练的时间
        start_time = time.time()

        writer = SummaryWriter('./logs_train')
        print('----------Start Training----------')
        for i in range(self.iteration_times):
            self.natural_select()
            self.cross()
            self.mutation()
            self.population = self.parents + self.childs

            # 每次迭代后仅计算一次最优路径
            new_path, new_dis = self.get_result()
            if new_dis < cur_dis:
                cur_path, cur_dis = new_path, new_dis

            # 添加数据到tensorboard
            # 数据量大，迭代100次输出一次
            if (i + 1) % 100 == 0:
                writer.add_scalar('Nicaragua_cur_path', new_dis, i + 1)
                print(f"Iteration {i + 1}/{self.iteration_times}: Current Min Distance = {cur_dis}")  # 添加进度输出

        print('----------End----------')
        writer.close()

        # 输出训练花费时间
        end_time = time.time()
        train_time = end_time - start_time
        print(f'Cost time: {train_time}')

        return cur_path, cur_dis


file_name = "nu3496.tsp"
modified_times = 3000
remain_rate = 0.5
select_rate = 0.3
count = 300
mutation_rate = 0.1
iteration_times = 5000


def main():
    search_TSP = GeneticAlgTSP(file_name, modified_times, remain_rate, select_rate, count, mutation_rate, iteration_times)
    min_path, min_dis = search_TSP.iterate()

    print(f"Min path: {min_path}, Min distance: {min_dis}.")


if __name__ == "__main__":
    main()
