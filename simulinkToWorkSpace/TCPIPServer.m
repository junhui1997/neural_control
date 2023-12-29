% 服务器代码，工作区这里必须要用两个服务器，因为simulink中的Receive和Send都是作为客户机存在的
close all; clear; clc

% 建立两个服务器，分别实现 ”向simulink中写入数据“ 和 ”从simulink中获取数据“
serverSend=tcpip('127.0.0.1',30000,'NetworkRole','Server','ByteOrder','littleEndian');
serverReceive=tcpip('127.0.0.1',30001,'NetworkRole','Server','ByteOrder','littleEndian');
disp("ready to connect");
serverSend.OutputBufferSize=100000;
serverReceive.OutputBufferSize=100000;
% 分别建立两个服务器与simulink的连接
fopen(serverSend);
disp("成功与Simulink中的Receive建立连接");
fopen(serverReceive);
disp("成功与Simulink中的Send建立连接");
disp(" ");

% 发送和接收sendData，sendData为1-10共10个数字
sendData=linspace(1,10,10);
for i=1:length(sendData)
    % 向simulink中写入第i个数字
    fwrite(serverSend,sendData(i),'double');
    disp("向simulink中写入了数据： "+sendData(i));
    % 从simulink中接收信息并显示，receiveData为接收到的信息
    while(1) 
        if serverReceive.BytesAvailable>0
            break;
        end
    end
    receiveData=fread(serverReceive,serverReceive.BytesAvailable/8,'double');
    disp("从simulink中获取了数据： "+receiveData);
end

%% 关闭服务器
fclose(serverSend);
fclose(serverReceive);
disp("关闭服务器");