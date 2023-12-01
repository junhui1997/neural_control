import matlab.engine
import numpy as np

def parse_out(to_parse):
    tp = type(to_parse)
    a = type(float(1.0))
    if type(to_parse) == float:
        return [to_parse]
    res = []
    to_parse = np.array(to_parse)
    for i in range(len(to_parse)):
        res.append(to_parse[i][0])
    return res

engine = matlab.engine.start_matlab()
print('sys start')
engine.neural(nargout=0)
# x = engine.eval('x')
# engine.workspace['x'] = x
# sys config
counter = 1
sample_rate = 0.2
total_t = 10
# 一定要注意data type问题
x = 1
while True:
    engine.set_param('example_0413', 'SimulationCommand', 'pause', nargout=0)
    # 在这里插入要更新的
    x+=1.0;
    engine.workspace['x'] = x
    sys_out = engine.eval('out.simout')
    print(sys_out)
    sys_out = parse_out(sys_out)
    print(sys_out)
    # status = engine.get_param('example_0413', 'SimulationStatus')
    # print(status)
    engine.set_param('example_0413', 'SimulationCommand', 'update', nargout=0)
    engine.set_param('example_0413', 'SimulationCommand', 'step', nargout=0)
    if counter >= total_t / sample_rate:
        engine.set_param('example_0413', 'SimulationCommand', 'stop', nargout=0)
        break
    counter = counter + 1
# engine.quit() # quit Matlab engine