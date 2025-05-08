import time
import os


def main(iters):
    for i in range(iters,iters+1):
        # 执行你的 Python 脚本，假设脚本名为 your_script.py
        os.system(f'python test_for_analysis.py --feature_id={i}')
        # 等待一段时间再继续执行，例如等待 60 秒
        # time.sleep(60)


if __name__ == "__main__":
    iters = 511
    main(iters)