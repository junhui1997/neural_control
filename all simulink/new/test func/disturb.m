t = 16.1;
if t>15 && t<21.2
% 仅HyOFC可以
%     d=[120.0*sin(1*(t-15));
%                          0;
%        100.0*sin(1*(t-15));
%                zeros(3,1)];
% M-BC和HyOFC都可以          
    d=[50.0*sin(1*(t-15));
                         0;
       15.0*sin(1*(t-15));
               zeros(3,1)];
else
    d=zeros(6,1);
end