
% 手动设置固定的输入值
W = ones(26, 1);  % 长度为 26，全为 1 的列向量
e13_com = 0.5; e14_com = 0.7;  % 设定 e13_com 和 e14_com 的值
qd3 = 0.3; qd4 = 0.4; dqd3 = 0.2; dqd4 = 0.1; ddqd3 = 0.6; ddqd4 = 0.8;  % 设定其他值
q = [0.1; 0.2];  % 长度为 2 的列向量
dq = [0.4; 0.6];  % 长度为 2 的列向量

%% 一、参数
%     1)控制器增益
            k13=15;k14=15;    k23=1;k24=1;    k33=100;k34=100;
            K1=[k13;k14];     K2=[k23;k24];   K3=[k33;k34];
            K1=diag(K1);      K2=diag(K2);    K3=diag(K3);
%     2)双环BLF
            rho_13=.008;rho_14=.008;  rho_33=.5;rho_34=.5;
            rho_11=rho_13;rho_12=rho_14;  rho_31=rho_33;rho_32=rho_34;
%     3)RBFNN            
            nodes=13;  Tau11=60;Tau12=60;  eta1=0.01;  b=60;
            c=[-30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30;
               -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30];
            
%% 二、系统输入
e1_com=[e13_com;e14_com];
qd=[qd3;qd4]; 
dqd=[dqd3;dqd4];
ddqd=[ddqd3;ddqd4];

%% 三、编码器
Delta=2*pi/2^19;
q_sample=[0;0];
for i=1:1:2
    Number=floor(q(i)/Delta)+1;
    q_sample(i)=Number*Delta;
end

%% 四、误差信号
e1=q_sample-qd;
% e1=q-(qd-.5*e1_com); % OK
de1=dq-dqd;

%% 五、虚拟控制器
miu=-K1*e1+dqd;
dmiu=-K1*de1+ddqd;

%% 六、辅助误差信号
e2=dq-miu;
e3=K2*e1+e2;

%% 七、双环BLF
%     1)外环
            u_BLF1=[e1(1)/(rho_11^2-e1(1)^2);
                    e1(2)/(rho_12^2-e1(2)^2)];     % u_BLF1：2行1列
%     2)内环
            u_BLF2_denominator_1=rho_31^2-e3(1)^2;
            u_BLF2_denominator_2=rho_32^2-e3(2)^2;
            u_BLF2=[e3(1)/u_BLF2_denominator_1;
                    e3(2)/u_BLF2_denominator_2];   % u_BLF2：2行1列
            Gain_3=diag([u_BLF2_denominator_1;u_BLF2_denominator_2]);

%% 八、RBFNN
%     1)RBF Kernel 
            xi1=[q_sample(1);dq(1)];xi2=[q_sample(2);dq(2)];
            h1=zeros(nodes,1);h2=zeros(nodes,1);
            for j=1:1:nodes
                h1(j)=exp(-norm(xi1-c(:,j))^2/(b*b));
                h2(j)=exp(-norm(xi2-c(:,j))^2/(b*b));
            end
%      2)estimate
            W1=W(1:nodes,1);W2=W(nodes+1:2*nodes,1);
            fn=[W1'*h1;W2'*h2];
%      3)dynamics of W
            dW=zeros(2*nodes,1);
            for i=1:1:nodes
                dW(i)=Tau11*h1(i)*u_BLF2(1)-eta1*W1(i); 
                dW(nodes+i)=Tau12*h2(i)*u_BLF2(2)-eta1*W2(i); 
            end

%% 九、真实控制器
%     1) 6行6列的模型矩阵
            M66 = [M(1) M(2) M(3) M(4) M(5) M(6)
                   M(7) M(8) M(9) M(10) M(11) M(12)
                   M(13) M(14) M(15) M(16) M(17) M(18)
                   M(19) M(20) M(21) M(22) M(23) M(24)
                   M(25) M(26) M(27) M(28) M(29) M(30)
                   M(31) M(32) M(33) M(34) M(35) M(36)];
%     2) 核心组件
            Torque_parts=-K2*de1-K3*e3-fn+dmiu-Gain_3*u_BLF1-diag([0.5;0.5])*u_BLF2; % Torque_parts: 2行1列
%     3) 六维输出
            Torque=M66*[0;0;Torque_parts;0;0];
% Torque_out_2=[Torque(3);Torque(4)];   

%% 十、输出
Out=[dW;Torque];  %  Torque_out_2
aaa = [12.36365447214695;
14.426161339360473;
16.371049935584224;
18.06873361923074;
19.395754889609513;
20.249528875830844;
20.56142641375376;
20.305884522318053;
19.503866477805406;
18.220021752962303;
16.554079609052007;
14.628065823487459;
12.571613702039869;
13.76112017801512;
16.069958320442797;
18.251516225951;
20.16088775590572;
21.65952822609866;
22.631759041742153;
22.99949762143537;
22.73261363345522;
21.85300602597073;
20.43163550045753;
18.579070304592136;
16.43129282178552;
14.133292492630224;
];
ares = dW-aaa;
sum(ares)