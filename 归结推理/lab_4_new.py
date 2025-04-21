import copy

# 逻辑公式类
class LogicFormula:
    def __init__(self, negation_flag, func_name, args):
        self.negation_flag = negation_flag
        self.func_name = func_name
        self.args = args

    def __repr__(self):
        representation = ''
        if self.negation_flag == 0:
            representation += '~'
        representation += self.func_name + '('
        representation += ','.join(self.args)
        representation += ')'
        return representation

# 归结操作
def perform_resolution(clause1, clause2, index1, index2, knowledge_base, step_counter, output):
    len1 = len(clause1)
    len2 = len(clause2)
    new_clause = []
    
    for i in range(len1):
        for j in range(len2):
            if clause1[i].negation_flag != clause2[j].negation_flag and clause1[i].func_name == clause2[j].func_name:
                if clause1[i].args == clause2[j].args:
                    new_clause = clause2[:j] + clause2[j + 1:] + clause1[:i] + clause1[i + 1:]
                    if is_clause_unique(new_clause, knowledge_base):
                        result_entry = f'{step_counter[0]} R[{index1}'
                        knowledge_base.append(new_clause)
                        step_counter[0] += 1

                        if len1 > 1:
                            result_entry += str(chr(i + 97))
                        result_entry += f',{index2}'

                        if len2 > 1:
                            result_entry += str(chr(j + 97))
                        result_entry += '] = '
                        result_entry += convert_clause_to_string(new_clause)
                        output.append(result_entry)

                        if new_clause == []:
                            return 0
                else:
                    unify_and_resolve(clause1, clause2, i, j, index1, index2, knowledge_base, step_counter, output)

    return 1

def unify_and_resolve(clause1, clause2, index1, index2, clause_index1, clause_index2, knowledge_base, step_counter, output):
    len1 = len(clause1[index1].args)
    
    for k in range(len1):
        if not is_variable(clause1[index1].args[k]) and is_variable(clause2[index2].args[k]):
            modified_clauses = copy.deepcopy(clause2)
            for t in range(len(modified_clauses)):
                for r in range(len(modified_clauses[t].args)):
                    if modified_clauses[t].args[r] == clause2[index2].args[k]:
                        modified_clauses[t].args[r] = clause1[index1].args[k]

            new_clause = modified_clauses[:index2] + modified_clauses[index2 + 1:] + clause1[:index1] + clause1[index1 + 1:]

            if is_clause_unique(new_clause, knowledge_base):
                knowledge_base.append(new_clause)
                result_entry = f'{step_counter[0]} R[{clause_index1}'
                step_counter[0] += 1

                if len(clause1) > 1:
                    result_entry += str(chr(index1 + 97))
                result_entry += f',{clause_index2}'
                if len(clause2) > 1:
                    result_entry += str(chr(index2 + 97))

                result_entry += f']{{{clause2[index2].args[k]} = {clause1[index1].args[k]}}}'
                result_entry += convert_clause_to_string(new_clause)
                output.append(result_entry)
                if new_clause == []:
                    return 0
        elif not is_variable(clause2[index2].args[k]) and is_variable(clause1[index1].args[k]):
            modified_clauses = copy.deepcopy(clause1)
            for t in range(len(modified_clauses)):
                for r in range(len(modified_clauses[t].args)):
                    if modified_clauses[t].args[r] == clause1[index1].args[k]:
                        modified_clauses[t].args[r] = clause2[index2].args[k]
            new_clause = clause2[:index2] + clause2[index2 + 1:] + modified_clauses[:index1] + modified_clauses[index1 + 1:]

            if is_clause_unique(new_clause, knowledge_base):
                knowledge_base.append(new_clause)
                result_entry = f'{step_counter[0]} R[{clause_index1}'
                step_counter[0] += 1

                if len(clause1) > 1:
                    result_entry += str(chr(index1 + 97))
                result_entry += f',{clause_index2}'
                if len(clause2) > 1:
                    result_entry += str(chr(index2 + 97))

                result_entry += f']{{{clause1[index1].args[k]} = {clause2[index2].args[k]}}}'
                result_entry += convert_clause_to_string(new_clause)
                output.append(result_entry)
                if new_clause == []:
                    return 0

# 将字符串转换为逻辑公式类
def convert_string_to_logic_formula(formula_str):
    length = len(formula_str)
    negation_flag = 1
    func_name = ''
    args = []
    if formula_str[0] == '~':
        negation_flag = 0
    func_started = 0
    start_index = 0
    for i in range(length):
        if formula_str[i] == '(':
            func_started = 1
            start_index = i + 1
            func_name = formula_str[1:i] if negation_flag == 0 else formula_str[0:i]

        if func_started == 1:
            if formula_str[i] == ',' or formula_str[i] == ')':
                args.append(formula_str[start_index:i])
                start_index = i + 1

    return LogicFormula(negation_flag, func_name, args)

# 解析为公式类
def convert_kb_to_logic_formula_list(knowledge_base):
    for i in range(len(knowledge_base)):
        for j in range(len(knowledge_base[i])):
            knowledge_base[i][j] = convert_string_to_logic_formula(knowledge_base[i][j])

# 将子句列表转换为字符串
def convert_clause_to_string(clauses):
    return str(tuple(clauses))

# 检查子句是否存在
def is_clause_unique(new_clause, knowledge_base):
    new_set = frozenset(str(x) for x in new_clause)
    for clause in knowledge_base:
        if frozenset(str(x) for x in clause) == new_set:
            return False
    return True

# 判断参数是否为变量（只包含小写字母）
def is_variable(param):
    return param.islower()

# 归结主函数
def ResolutionFOL(knowledge_base):
    output = []
    knowledge_base = list(knowledge_base)

    for i in range(len(knowledge_base)):
        knowledge_base[i] = list(knowledge_base[i])
    convert_kb_to_logic_formula_list(knowledge_base)
    
    step_counter = [1]

    for i in range(len(knowledge_base)):
        initial_entry = f'{step_counter[0]} {convert_clause_to_string(knowledge_base[i])}'
        output.append(initial_entry)
        step_counter[0] += 1
    
    # 穷举归结
    for k in range(len(knowledge_base)):
        for i in range(len(knowledge_base)):
            for j in range(i + 1, len(knowledge_base)):
                resolution_result = perform_resolution(knowledge_base[i], knowledge_base[j], i + 1, j + 1, knowledge_base, step_counter, output)
                if resolution_result == 0:
                    return output

# 示例1
print("示例1")
knowledge_base1 = {("GradStudent(Sue)",),
                    ("~GradStudent(x)", "Student(x)"),
                    ("~Student(x)", "HardWorker(x)"),
                    ("~HardWorker(Sue)",)}
result1 = ResolutionFOL(knowledge_base1)
for entry in result1:
    print(entry)

# 示例2
print("示例2")
knowledge_base2 = {("On(Tony,Mike)",), ("On(Mike,John)",), ("Green(Tony)",), ("~Green(John)",), ("~On(x,y)", "~Green(x)", "Green(y)")}
result2 = ResolutionFOL(knowledge_base2)
for entry in result2:
    print(entry)
