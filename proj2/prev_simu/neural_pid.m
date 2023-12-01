Code_2DoF_Simulation
close all; clear; clc
o_py = [0.0,0.0];
set_param('Code_2DoF_Simulation','SimulationCommand','stop'); %stop是终止之前的simulation，重新开启一次新的
set_param('Code_2DoF_Simulation','SimulationCommand','start');
set_param('Code_2DoF_Simulation','SimulationCommand','pause');