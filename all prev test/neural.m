example_0413
close all; clear; clc
x = 1;
set_param('example_0413','SimulationCommand','stop'); %stop是终止之前的simulation，重新开启一次新的
set_param('example_0413','SimulationCommand','start');
set_param('example_0413','SimulationCommand','pause');