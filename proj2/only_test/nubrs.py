from geomdl import NURBS
from geomdl import utilities
import numpy as np

# 定义控制点和权重
control_points = [[0, 0], [1, 2], [2, -1], [3, 3]]  # 示例控制点
weights = [1, 1, 1,  1]  # 示例权重，所有都设为1

# 创建NURBS曲线实例
curve = NURBS.Curve()

# 设置曲线的度数
curve.degree = 3  # 3次B样条曲线

# 加载控制点和权重
curve.ctrlpts = list(map(lambda cp, w: [*cp, w], control_points, weights))

# 自动创建曲线knots向量
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(control_points))

# 定义时间范围
t = np.linspace(0, 1, 2000)  # 从0到1的2000个均匀分布的时间点

# 使用时间参数t生成曲线上的点
trajectory_points = np.array([curve.evaluate_single(ti) for ti in t])

# 去掉权重维度，只保留x, y轴数据
trajectory_points = trajectory_points[:, :2]

# 现在 trajectory_points 是机械臂轴的轨迹点，你可以根据需要对其进行操作或绘图
import matplotlib.pyplot as plt

# 绘制NURBS曲线
plt.figure()
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], label="NURBS Trajectory")

# 标注图形
plt.title('NURBS Curve Trajectory')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
