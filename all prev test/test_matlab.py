import matlab.engine
engine = matlab.engine.start_matlab()
engine.example(nargout=0) # 通过eng运行写好的m文件 “example.m”，nargout=0表示不返回输出
for i in range(5):
    clock = engine.test(float(i+1), nargout=1)
    print(clock)
engine.quit() # quit Matlab engine