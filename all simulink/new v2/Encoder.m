q = [0.1; 0.2];
Delta =2*pi/2^19;
q_sample=[0;0];
for i=1:1:2
    Number=floor(q(i)/Delta)+1;
    q_sample(i)=Number*Delta;
end
aaa = [0.1000083568352006;
0.20000472944549583;];
q_sample - aaa