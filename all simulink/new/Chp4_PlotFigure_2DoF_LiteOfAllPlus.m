close all;
BLFmax=0.15;BLFmin=0.01;BLFrate=0;  % 0.03
min=.008;max=.008;B_1_U=max;B_1_L=-min;B_2_U=min;B_2_L=-max;B_3_U=max;B_3_L=-min;
                B_4_U=max;B_4_L=-min;B_5_U=max;B_5_L=-min;B_6_U=max;B_6_L=-min;
nodes=13;
FinalTime=30;
NNstep=5;
i=0:1:FinalTime*400;
t1=0:0.0025:FinalTime;t1=t1';

% SSIdq=SS(1+NNstep*i,19:24);
% Joint Angle
figure(1);
subplot(221);
plot(tout,VRRef(:,1),'r',tout,q(:,1),'k','linewidth',1.0);hold on;
plot(tout,(VRRef(:,1)+B_3_U),'Color',[0.93 0.69 0.13],'linewidth',1.0);hold on;
plot(tout,(VRRef(:,1)+B_3_L),'Color',[0 0.45 0.74],'linewidth',1.0);hold on;
xlabel('Time (s)');ylabel('\itq\rm_{\rm3} (rad)'); 
subplot(222);
plot(tout,VRRef(:,2),'r',tout,q(:,2),'k','linewidth',1.0);hold on;
plot(tout,(VRRef(:,2)+B_4_U),'Color',[0.93 0.69 0.13],'linewidth',1.0);hold on;
plot(tout,(VRRef(:,2)+B_4_L),'Color',[0 0.45 0.74],'linewidth',1.0);hold on;
ylabel('\itq\rm_{\rm4} (rad)');
  
subplot(223);
plot(tout,VRRef(:,1),'r',tout,q(:,1),'k','linewidth',1.0);hold on;
ylabel('\itq\rm_{\rm1} (rad)');
subplot(224);
plot(tout,VRRef(:,2),'r',tout,q(:,2),'k','linewidth',1.0);hold on;
ylabel('\itq\rm_{\rm2} (rad)');

% Tracking error of Joint Angle
figure(2);
DEG=180/pi;
% DEG=1;
Ymax=0.1;
subplot(221);
plot(tout,abs(-VRRef(:,1)+q(:,1)),'k','linewidth',1.0);hold on;
plot(t1,0*t1+B_4_U,'Color',[0.93 0.69 0.13],'linewidth',1.0);hold on;
plot(t1,0*t1+B_4_L,'Color',[0 0.45 0.74],'linewidth',1.0);hold on;
ylabel('\itq\rm_{\ite\rm1} (rad)'); 
subplot(222);
plot(tout,abs(-VRRef(:,2)+q(:,2)),'k','linewidth',1.0);hold on;
plot(t1,0*t1+B_4_U,'Color',[0.93 0.69 0.13],'linewidth',1.0);hold on;
plot(t1,0*t1+B_4_L,'Color',[0 0.45 0.74],'linewidth',1.0);hold on;
ylabel('\itq\rm_{\ite\rm2} (rad)');
subplot(223);
plot(tout,abs(-VRRef(:,2)+q(:,2))*DEG,'color',[.64 .08 .18],'linewidth',1.0);hold on;
plot(tout,abs(-VRRef(:,1)+q(:,1))*DEG,'k','linewidth',1.0);hold on;
ylabel('\itq\rm_{\ite\rm1} (deg)'); 
subplot(224);
plot(tout,abs(-VRRef(:,2)+q(:,2))*DEG,'k','linewidth',1.0);hold on; % axis([0 FinalTime 0 Ymax]); 
ylabel('\itq\rm_{\ite\rm2} (deg)'); 

% Joint Velocity
figure(3);
subplot(321);
plot(tout,ddqSat(:,1),'k','linewidth',1.0);hold on;  
ylabel('dd\itq\rm_{\rm1} (rad/s/s)');
subplot(322);
plot(tout,ddqSat(:,2),'k','linewidth',1.0);hold on; 
ylabel('dd\itq\rm_{\rm2} (rad/s/s)');
subplot(323);
plot(tout,VRRef(:,3),'r',tout,dq(:,1),'k','linewidth',1.0);hold on;  
ylabel('d\itq\rm_{\rm1} (rad/s)');
subplot(324);
plot(tout,VRRef(:,4),'r',tout,dq(:,2),'k','linewidth',1.0);hold on; 
ylabel('d\itq\rm_{\rm2} (rad/s)');
Ymax=0.01;
subplot(325);
plot(tout,abs(VRRef(:,4)-dq(:,2)),'color',[.64 .08 .18],'linewidth',1.0);hold on; 
plot(tout,abs(VRRef(:,3)-dq(:,1)),'k','linewidth',1.0);hold on; % axis([0 FinalTime 0 Ymax]);
ylabel('d\itq\rm_{\rm1} (rad/s)');
subplot(326);
plot(tout,abs(VRRef(:,4)-dq(:,2)),'k','linewidth',1.0);hold on; % axis([0 FinalTime 0 Ymax]); 
ylabel('d\itq\rm_{\rm2} (rad/s)');

% figure(5);
% subplot(221);
% plot(t1,fnReal(:,3),'r',t1,fn(:,1),'k','linewidth',1.0);hold on; 
% ylabel('fn3'); 
% subplot(222);
% plot(t1,fnReal(:,4),'r',t1,fn(:,2),'k','linewidth',1.0);hold on;
% ylabel('fn4'); 

% Joint Torque
figure(6);
subplot(221);
plot(t1,Tau(:,3),'k','linewidth',1.0);hold on;
ylabel('\tau_1 (Nm)'); 
subplot(222);
plot(t1,Tau(:,4),'k','linewidth',1.0);hold on; 
ylabel('\tau_2 (Nm)'); 

% figure(8);
% for i=1:length(tout)
%    q3=q(i,1);q4=q(i,2);
%    q3_Ref=VRRef(i,1);q4_Ref=VRRef(i,2);
%    n=show_plot_sfijdl(0,pi/2,q3,q4,0,0);
%    n_Ref=show_plot_sfijdl(0,pi/2,q3_Ref,q4_Ref,0,0);
%    EEx(i)=n(1);EEy(i)=n(2);EEz(i)=n(3);
%    EEx_Ref(i)=n_Ref(1);EEy_Ref(i)=n_Ref(2);EEz_Ref(i)=n_Ref(3);
% end
% Err=sqrt((EEx_Ref-EEx).^2+(EEz_Ref-EEz).^2);
% subplot(121);
% % plot3(EEx_Ref,EEy_Ref,EEz_Ref,'Color',[1 0 0],'linewidth',1.5);hold on;
% % plot3(EEx,EEy,EEz,'Color',[0 0 0],'linewidth',1.5);hold on;
% plot(EEx_Ref,EEz_Ref,'k','linewidth',1.5);hold on;
% plot(EEx,EEz,'g--','linewidth',1.5);hold on;axis equal;
% subplot(122);
% plot(tout,Err,'k','linewidth',1.0);hold on;