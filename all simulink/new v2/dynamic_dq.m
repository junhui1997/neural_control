ddq_max = 100*pi;
dq_Max = 100*pi;
q = [0.1; 0.2];
dq = [0.4; 0.6];
ddq = [0.44; 0.64];
Torque = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6];
z = [0.1; 0.2; 0.7; 0.8; 0.5; 0.6];
dt = 0.123;
dq=dq+ddq*dt;
for i=1:1:2
    if dq(i)>dq_Max
        dq(i)=dq_Max;
    elseif dq(i)<-dq_Max
        dq(i)=-dq_Max;
    end
end
dq_out=[dq(1);dq(2)];
aaa = [0.45412;
0.67872;];
dq_out - aaa