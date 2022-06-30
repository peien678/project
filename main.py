# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import feature_stats
import pandas as pd
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    import h5py
    # 安装h5py库,导入
    f = h5py.File('D:/data/20220422_binanceUsdtSwap_btc-usdt_depth.h5', 'r')
    # 读取文件,一定记得加上路径

    for key in f.keys():
        print(key)
        # print(f[key].name)
        # 打印出文件中的关键字
        print(f[key].shape)
        # # 将key换成某个文件中的关键字,打印出某个数据的大小尺寸
        # print(f[key][:])
        # # 将key换成某个文件中的关键字,打印出该数据(数组)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
