Code_2DoF_Simulation
close all; clear; clc
o_py = [0.0,0.0];
set_param('Code_2DoF_Simulation','SimulationCommand','stop'); %stop是终止之前的simulation，重新开启一次新的
set_param('Code_2DoF_Simulation','SimulationCommand','start'); %直接start之后，如果没有pause就会直接执行到任务的结尾
set_param('Code_2DoF_Simulation','SimulationCommand','pause');
counter = 1;
sample_rate = 0.0025;
total_t = 32;

server = tcpip('127.0.0.1', 30000, 'NetworkRole', 'Server','ByteOrder','littleEndian'); %littleEndian
% 等待客户端连接
server.OutputBufferSize=100000;
disp('等待客户端连接...');
fopen(server);
disp('客户端已连接');


while true
    if server.BytesAvailable>0
        set_param('Code_2DoF_Simulation','SimulationCommand','pause');
        to_send = [e1(end,1),e1(end,2),q(end,1),q(end,2),sTau(end,1),sTau(end,2),dq(end,1),dq(end,2)];
        to_send(1) = 1+counter;
        fwrite(server,to_send,'double')
        reveive_data = fread(server, 2,'double');
        x = reveive_data(1);
        set_param('Code_2DoF_Simulation','SimulationCommand','update');
        set_param('Code_2DoF_Simulation','SimulationCommand','step'); %step就是单纯的步进一步不需要额外再pause了
        counter = counter+1;
        if counter >= total_t/sample_rate
            set_param('Code_2DoF_Simulation','SimulationCommand','stop');
            break;
        end
    end
end