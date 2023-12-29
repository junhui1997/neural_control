clear
% 创建Server Socket
s = tcpip('127.0.0.1', 30000, 'NetworkRole', 'Server','ByteOrder','littleEndian'); %littleEndian
% 等待客户端连接
s.OutputBufferSize=100000;
disp('等待客户端连接...');
fopen(s);
disp('客户端已连接');



counter = 0;
to_send = [0.1,0.2,0.3];
while true 
    %counter = counter +1
    %s.BytesAvailable
    if s.BytesAvailable>0
        B = fread(s, 3,'double');
        fwrite(s,to_send,'double')
%         receive = fscanf(s);
        counter = counter +1
    end
end

% %发送path
% function sendPath(s)
%         path=rand(4,10)
%         path_bytes=reshape(path,[40,1])
%         fwrite(s,'path','char')
%         fwrite(s,path_bytes,'double')
% end
% 
% %发送U
% function sendU(s)
%         u=rand(1,1);
%         fwrite(s,'uuuu','char')
%         fwrite(s,u,'double')
% end