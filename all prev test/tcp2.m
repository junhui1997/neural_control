ip = '127.0.0.1';
port = 30000;
% 构造服务器端tcpip对象
server = tcpserver(ip, port,"Timeout",40);
server.ConnectionChangedFcn = @requestDataCommand;
configureCallback(server,"terminator", @readArduinoData);
configureTerminator(server,"LF");

function requestDataCommand(src,~)
    if src.Connected
        % Display the server object to see that Arduino client has connected to it.
        disp("The Connected and ClientAddress properties of the tcpserver object show that the Arduino is connected.")
        %       disp(src.BytesAvailableFcnMode)
        % Request the Arduino to send data.
        disp("Send the command: ")
    end
end

function readArduinoData(src,~)
	% Read the sine wave data sent to the tcpserver object.
	recive_tmie = tic;
	src.UserData = readline(src);
	input = jsondecode(src.UserData);
	toc(recive_tmie)
end
