function [   weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,bias_gate,bias_input_gate,bias_forget_gate,bias_output_gate]=LSTM_updata_weight_bias(cell_num,output_num,n,yita,Error,...
                                                   weight_input_x, weight_input_h, weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,bias_gate,bias_input_gate,bias_forget_gate,bias_output_gate,.....
                                                   cell_state,h_state,input_gate,forget_gate,output_gate,gate,train_data,pre_h_state,input_gate_input,output_gate_input,forget_gate_input)
%%% 权重更新函数
data_length=size(train_data,1);
data_num=size(train_data,2);
weight_preh_h_temp=weight_preh_h;

%% 更新weight_preh_h权重
for m=1:output_num
    delta_weight_preh_h_temp(:,m)=-Error(m,1)*pre_h_state(:,n);
end
weight_preh_h=weight_preh_h-yita*delta_weight_preh_h_temp;

if n==1
    %% 更新weight_input_x
%     temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(gate))-gate.^2)*train_data(m,n);
    end
    weight_input_x=weight_input_x-yita*delta_weight_input_x;
    end

%% 更新weight_inputgate_x
for num=1:output_num
for m=1:data_length
    delta_weight_inputgate_x(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n);
end
weight_inputgate_x=weight_inputgate_x-yita*delta_weight_inputgate_x;
end

%% 更新weight_outputgate_x
for num=1:output_num
    for m=1:data_length
        delta_weight_outputgate_x(m,:)=(-weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2)*train_data(m,n);
    end
    weight_outputgate_x=weight_outputgate_x-yita*delta_weight_outputgate_x;
end

%% 更新bias_gate
for num=1:output_num
    delta_bias_gate=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(gate))-gate.^2);
bias_gate=bias_gate-yita*delta_bias_gate;
end

%% 更新bias_input_gate
for num=1:output_num
    delta_bias_input_gate=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2);
bias_input_gate=bias_input_gate-yita*delta_bias_input_gate;
end

%% 更新bias_output_gate
for num=1:output_num
    delta_bias_output_gate=(-weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2);
bias_output_gate=bias_output_gate-yita*delta_bias_output_gate;
end

else
    %% 更新weight_input_x
%     temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(gate))-gate.^2)*train_data(m,n);
    end
    weight_input_x=weight_input_x-yita*delta_weight_input_x;
    end
    
%% 更新weight_inputgate_x
for num=1:output_num
for m=1:data_length
    delta_weight_inputgate_x(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n);
end
weight_inputgate_x=weight_inputgate_x-yita*delta_weight_inputgate_x;
end
    
    %% 更新weight_forgetgate_x
    for num=1:output_num
    for m=1:data_length
        delta_weight_forgetgate_x(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*train_data(m,n);
    end
    weight_forgetgate_x=weight_forgetgate_x-yita*delta_weight_forgetgate_x;
    end
    
%% 更新weight_outputgate_x
for num=1:output_num
    for m=1:data_length
        delta_weight_outputgate_x(m,:)=(-weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2)*train_data(m,n);
    end
    weight_outputgate_x=weight_outputgate_x-yita*delta_weight_outputgate_x;
end    
    
 %% 更新weight_input_h
%     temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:output_num
        delta_weight_input_h(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(gate))-gate.^2)*pre_h_state(m,n-1);
    end
    weight_input_h=weight_input_h-yita*delta_weight_input_h;
    end

    %% 更新weight_inputgate_h
    for num=1:output_num
    for m=1:cell_num
        delta_weight_inputgate_c(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*pre_h_state(m,n-1);
    end
    weight_inputgate_c=weight_inputgate_c-yita*delta_weight_inputgate_c;
    end
    %% 更新weight_forgetgate_h
    for num=1:output_num
    for m=1:cell_num
        delta_weight_forgetgate_c(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*pre_h_state(m,n-1);
    end
    weight_forgetgate_c=weight_forgetgate_c-yita*delta_weight_forgetgate_c;
    end
    %% 更新weight_outputgate_h
    for num=1:output_num
    for m=1:cell_num
        delta_weight_outputgate_c(m,:)=-(weight_preh_h(:,num)*Error(num,1))'.*tanh(cell_state(:,n))'.*exp(-output_gate_input).*(output_gate.^2)*pre_h_state(m,n-1);
    end
    weight_outputgate_c=weight_outputgate_c-yita*delta_weight_outputgate_c;
    end
    
%% 更新bias_gate
for num=1:output_num
    delta_bias_gate=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(gate))-gate.^2);
bias_gate=bias_gate-yita*delta_bias_gate;
end

%% 更新bias_input_gate
for num=1:output_num
    delta_bias_input_gate=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2);
bias_input_gate=bias_input_gate-yita*delta_bias_input_gate;
end

%% 更新bias_forget_gate
for num=1:output_num
    delta_bias_forget_gate=-(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2);
bias_forget_gate=bias_forget_gate-yita*delta_bias_forget_gate;
end

%% 更新bias_output_gate
for num=1:output_num
    delta_bias_output_gate=(-weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2);
bias_output_gate=bias_output_gate-yita*delta_bias_output_gate;
end

end

