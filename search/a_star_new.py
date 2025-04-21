# 一维存储
import heapq
import numpy


class Node:
    def __init__(self, state, g=0, parent=None):
        self.state = state
        self.g = g
        self.h = get_h(state)
        self.parent = parent

    def __lt__(self, other):
        if (self.g + self.h) == (other.g + other.h):
            return self.h < other.h
        return (self.g + self.h) < (other.g + other.h)


def get_h(state):
    h1 = 0
    h2 = 0

    for idx in range(16):
        cur = state[idx]

        if cur == 0:
            continue

        goal_idx = cur - 1
        goal_row = goal_idx // 4
        goal_col = goal_idx % 4

        cur_row = idx // 4
        cur_col = idx % 4

        # 曼哈顿距离
        h1 += abs(cur_row - goal_row) + abs(cur_col - goal_col)

        # 线性冲突法

        # 行冲突
        if cur_row == goal_row:
            # 检查该行后续元素的目标行是否在cur左边
            for i in range(idx + 1, (cur_row + 1) * 4):
                neibor = state[i]

                if neibor == 0:
                    continue

                neibor_goal_row = (neibor - 1) // 4
                neibor_goal_col = (neibor - 1) % 4

                if neibor_goal_row == cur_row and neibor_goal_col < goal_col:
                    h2 += 2

        # 列冲突
        if cur_col == goal_col:
            # 检查该行后续元素的目标行是否在cur左边
            for i in range(idx + 4, 16, 4):
                neibor = state[i]

                if neibor == 0:
                    continue

                neibor_goal_row = (neibor - 1) // 4
                neibor_goal_col = (neibor - 1) % 4

                if neibor_goal_col == cur_col and neibor_goal_row < goal_row:
                    h2 += 2

    return h1 + h2


def a_star(start):
    start = numpy.array(start)
    start = start.flatten()
    start = tuple(start)
    
    open_list = []
    close_list = set()
    heapq.heappush(open_list, Node(start))
    
    while open_list:
        node = heapq.heappop(open_list)

        # 检查是否达到目标状态
        if node.state == GOAL:
            output_path = []
            while node.parent:
                zero_prev = node.parent.state.index(0)
                moved_num = node.state[zero_prev]
                output_path.append(int(moved_num))
                node = node.parent
            # 反转路径
            return output_path[::-1]
        
        # 检查node是否在关闭列表中
        if node.state in close_list:
            continue
        close_list.add(node.state)
        
        zero_idx = node.state.index(0)
        row, col = zero_idx // 4, zero_idx % 4
        # 向四个方向移动
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dx, col + dy

            if 0 <= new_row <= 3 and 0 <= new_col <= 3:
                new_pos = new_row * 4 + new_col
                state = list(node.state)

                state[zero_idx], state[new_pos] = state[new_pos], state[zero_idx]
                # 将列表转换为元组以便于哈希
                new_state = tuple(state)

                if new_state not in close_list:
                    heapq.heappush(open_list, Node(new_state, node.g+1, node))
                    
    return None


# 示例调用
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

# 示例一
ex1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
path = a_star(ex1)

print("example_1 : ", path)

# 示例二
ex2 = [[1, 2, 4, 8], [5, 7, 11, 10], [13, 15, 0, 3], [14, 6, 9, 12]]
path = a_star(ex2)

print("example_2 : ", path)
