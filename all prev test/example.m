
example_0413
close all; clear; clc
x = 1;
% set_param('example_0413/Data Store Memory','InitialValue', num2str(x));
set_param('example_0413','SimulationCommand','stop'); %stop是终止之前的simulation，重新开启一次新的
set_param('example_0413','SimulationCommand','start'); %直接start之后，如果没有pause就会直接执行到任务的结尾
counter = 1;
sample_rate = 0.2;
total_t = 100;
while 1
set_param('example_0413','SimulationCommand','pause');
x = x+1;
set_param('example_0413','SimulationCommand','update');
set_param('example_0413','SimulationCommand','step'); 
%step就是单纯的步进一步不需要额外再pause了
%set_param('example_0413','SimulationCommand','pause');
counter = counter+1
if counter >= total_t/sample_rate
    set_param('example_0413','SimulationCommand','stop');
    break;
end
end
final_out =evalin('caller',' out.simout') ;
% assignin( base .clock',clock):