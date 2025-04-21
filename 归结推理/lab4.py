import copy

# 公式类
class formula:
    def __init__(self, ifnot, predicate, parameter):
        self.ifnot = ifnot  # 0表示否定,1表示肯定
        self.predicate = predicate  # 谓词
        self.parameter = parameter  # 参数列表

    def __repr__(self):
        str_prt = ''
        if self.ifnot == 0:
            str_prt += '~'
        str_prt += self.predicate + '('
        str_prt += ','.join(self.parameter) + ')'
        return str_prt

def is_variable(param):
    """判断是否为变量(只包含小写字母)"""
    return param.islower()

def unify(param1, param2):
    """合一算法 - 只允许变量替换为常量"""
    if param1 == param2:
        return None
    
    # 如果param1是变量,param2是常量,则可以替换
    if is_variable(param1) and not is_variable(param2):
        return (param1, param2)
    # 如果param2是变量,param1是常量,则可以替换    
    elif is_variable(param2) and not is_variable(param1):
        return (param2, param1)
    # 其他情况都不能合一    
    else:
        return None

def resolution(list1, list2, list1_index, list2_index, KB, step, result):
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i].ifnot != list2[j].ifnot and list1[i].predicate == list2[j].predicate:
                # 检查是否可以合一
                subst = []
                can_unify = True
                
                for k in range(len(list1[i].parameter)):
                    unify_result = unify(list1[i].parameter[k], list2[j].parameter[k])
                    if unify_result:
                        subst.append(unify_result)
                    elif list1[i].parameter[k] != list2[j].parameter[k]:
                        can_unify = False
                        break
                        
                if can_unify:
                    # 创建新子句并应用替换
                    new_list = []
                    # 复制并替换list1中的公式
                    for x in range(len(list1)):
                        if x != i:
                            f = copy.deepcopy(list1[x])
                            for var, const in subst:
                                f.parameter = [const if p == var else p for p in f.parameter]
                            new_list.append(f)
                            
                    # 复制并替换list2中的公式    
                    for x in range(len(list2)):
                        if x != j:
                            f = copy.deepcopy(list2[x])
                            for var, const in subst:
                                f.parameter = [const if p == var else p for p in f.parameter]
                            new_list.append(f)
                            
                    if not new_list in KB:
                        KB.append(new_list)
                        # 构造结果字符串
                        result_str = f"{step[0]} R[{list1_index}"
                        if len(list1) > 1:
                            result_str += chr(ord('a') + i)
                        result_str += f",{list2_index}"
                        if len(list2) > 1:
                            result_str += chr(ord('a') + j)
                        if subst:
                            result_str += "]{"
                            result_str += ",".join(f"{var}={const}" for var,const in subst)
                            result_str += "}"
                        else:
                            result_str += "]"
                        result_str += f" = {clause_to_str(new_list)}"
                        result.append(result_str)
                        step[0] += 1
                        
                        if not new_list:  # 空子句
                            return 0
    return 1

def parse_literal(literal):
    """解析原子公式，返回谓词和参数列表"""
    if literal.startswith('~'):
        pred = literal[1:literal.index('(')]
        is_neg = True
    else:
        pred = literal[:literal.index('(')]
        is_neg = False
    args = literal[literal.index('(')+1:literal.rindex(')')].split(',')
    return pred, args, is_neg

def clause_to_str(clause):
    if not clause:
        return "[]"
    return f"({','.join(clause)})"

def substitute(literal, subst):
    """应用替换"""
    pred, args, is_neg = parse_literal(literal)
    new_args = [subst.get(arg, arg) for arg in args]
    prefix = '~' if is_neg else ''
    return f"{prefix}{pred}({','.join(new_args)})"

def ResolutionFOL(KB):
    KB = list(KB)  # 转换成列表形式
    step = [1]     # 步骤计数器
    result = []    # 存储推理过程
    
    # 记录初始子句
    for i, clause in enumerate(KB):
        result.append(f"{step[0]} {clause}")
        step[0] += 1
    
    # 转换KB中的子句为formula对象
    for i in range(len(KB)):
        KB[i] = [str_to_class(literal) for literal in KB[i]]
    
    # 对每对子句进行归结
    for i in range(len(KB)):
        for j in range(i + 1, len(KB)):
            res = resolution(KB[i], KB[j], i + 1, j + 1, KB, step, result)
            if res == 0:  # 得到空子句
                return result
                
    return result

def str_to_class(literal):
    """将字符串形式的公式转换为formula对象"""
    is_neg = literal.startswith('~')
    if is_neg:
        pred = literal[1:literal.index('(')]
    else:
        pred = literal[:literal.index('(')]
        
    params = literal[literal.index('(')+1:literal.rindex(')')].split(',')
    return formula(0 if is_neg else 1, pred, params)

# 测试代码
if __name__ == "__main__":
    # 测试用例1
    KB1 = {("GradStudent(Sue)",),
           ("~GradStudent(x)","Student(x)"),
           ("~Student(x)","HardWorker(x)"),
           ("~HardWorker(Sue)",)}
    
    print("测试用例1:")
    steps = ResolutionFOL(KB1)
    for step in steps:
        print(step)
        
    # 测试用例2
    KB2 = {("On(Tony,Mike)",),("On(Mike,John)",),("Green(Tony)",),("~Green(John)",),("~On(x,y)","~Green(x)","Green(y)")}
    
    print("\n测试用例2:")
    result2 = ResolutionFOL(KB2)
    for step in result2:
        print(step)