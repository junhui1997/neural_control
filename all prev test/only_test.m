
Code_2DoF_Simulation
close all; clear; clc
x = 1;
o_py = [0,0];
% set_param('example_0413/Data Store Memory','InitialValue', num2str(x));
set_param('Code_2DoF_Simulation','SimulationCommand','stop'); %stop是终止之前的simulation，重新开启一次新的
set_param('Code_2DoF_Simulation','SimulationCommand','start'); %直接start之后，如果没有pause就会直接执行到任务的结尾
counter = 1;
sample_rate = 0.2;
total_t = 1000;
while 1
set_param('Code_2DoF_Simulation','SimulationCommand','pause');
x = x+1;
input_var = 200;
%current_out = extract_o;
set_param('Code_2DoF_Simulation','SimulationCommand','update');
set_param('Code_2DoF_Simulation','SimulationCommand','continue'); %step就是单纯的步进一步不需要额外再pause了
%set_param('example_0413','SimulationCommand','pause');
counter = counter+1
if counter >= total_t/sample_rate
    set_param('Code_2DoF_Simulation','SimulationCommand','stop');
    break;
end
end
% assignin( base .clock',clock):