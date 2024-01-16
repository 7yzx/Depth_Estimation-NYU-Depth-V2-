

def infer_type(x):  # hacky way to infer type from string args
    """
    用法：该函数尝试将输入值从字符串转换为整数或浮点数，并返回相应的类型。
    如果无法成功转换，则返回原始字符串。
    """
    if not isinstance(x, str):
        return x

    try:
        x = int(x)
        return x
    except ValueError:
        pass

    try:
        x = float(x)
        return x
    except ValueError:
        pass

    return x


def parse_unknown(unknown_args):
    """
    这个函数的目的是将未知参数列表转换为字典，
    其中键是参数名称（去除了可能存在的双横杠 "--"），
    而值是通过 infer_type 推断的参数值的类型。
    """
    clean = []
    for a in unknown_args:
        if "=" in a:
            k, v = a.split("=")
            clean.extend([k, v])
        else:
            clean.append(a)

    keys = clean[::2]
    values = clean[1::2]
    return {k.replace("--", ""): infer_type(v) for k, v in zip(keys, values)}
