function [clock]=test(action)
    set_param('example_0413/Gain','Gain', num2str(action));
    set_param('example_0413','SimulationCommand','step');
    clock=evalin('caller','out.simout (end, :)');
    %assignin(' base ,'clock'，clock):
end