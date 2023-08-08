%CASE1 SOLSTMԤ��
%���� �｣
%ʱ�� 2023��8��8��

%% ���
tic;
clc
clear;
close all;

%�̶��������,ʹ��ѵ��������� 
setdemorandstream(pi);
data_num=2000;
%���������ź�
for i=1:(data_num+10)
           u(i)=2*(-1+2*rand);
%            u(i)=1.75*sin(pi/25*i);
end     
y=zeros(1,data_num+10);    
d = 7;
for k=9:1:(data_num+10)
   y(k)=y(k-1)^3-0.2*abs(y(k-1))*u(k-d)+0.08*u(k-d)^2;
end
Xtrain=[y(9:1608);u(3:1602)]; 
Ytrain=y(10:1609);
[InDim,TrainSamNum]=size(Xtrain); %InDim����ά��4��TrainSamNumѵ��������500 4*500
OutDim=size(Ytrain,1); % OutDim���ά��Ϊ1
% ��������
Xtest=[y(1609:2008);u(1603:2002)];
Ytest=y(1610:2009);
TestSamNum=size(Xtest,2); %TestSamNum����������50

%LSTM model
%ѵ��������ǰ70%
input_train = Xtrain;%ѵ����������
output_train = Ytrain;%ѵ���������
%���Լ�������30%
input_test = Xtest;%������������
output_test = Ytest;%�����������

%���ݹ�һ����ͳһ����������λ
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
tn=mapminmax('apply',input_test,inputps);%���ݹ�һ��
yn=output_test;

%%%%%%%%%%%%%%%%
XTrain = inputn;
YTrain = outputn;
XTest = tn;
YTest = yn;

train_data = XTrain;
train_label = YTrain;
P_test = XTest;
T_test = YTest;

[InDim,train_num]=size(train_data);

% ѵ������
iter_num=100;

% % SOLSTM ------------------------
% % �̶��������,ʹ��ѵ��������� 
setdemorandstream(pi);

train_data = XTrain;
train_label = YTrain;
P_test = XTest;
T_test = YTest;

[data_length,train_num]=size(train_data);
% ����minibatch��������ѵ�� ������Ҫ��ѵ�����ݽ��з���
%��numbatches�� ÿ��data_num�� ��numbatches*data_num������
numbatches=1;
data_num=train_num;
for i=1:numbatches
    train1=train_data(:,(i-1)*data_num+1:i*data_num);
    batchdata(:,:,i)=train1;
end

for i=1:numbatches
    train2=train_label(:,(i-1)*data_num+1:i*data_num);
    batchdata_target(:,:,i)=train2;
end
%% ���������ʼ��
% ���������
input_num=data_length;%�����ڵ�
cell_num=20;%������ڵ�
output_num=1;%�����ڵ�
dropout=0;%dropoutϵ��
cost_gate=1e-10;% ���Ҫ�󾫶�
 ab=10;
step_window=10;

MASK=ones(1,cell_num);
% B=randperm(numel(MASK),0.2*cell_num);
% MASK(B)=0;

%% �����ʼ��
% ����ƫ�ó�ʼ��
bias_gate=1*rand(1,cell_num).*MASK;
bias_input_gate=1*rand(1,cell_num).*MASK;
bias_forget_gate=1*rand(1,cell_num).*MASK;
bias_output_gate=1*rand(1,cell_num).*MASK;
bias_ym_input=rand(1,1)*MASK;
% ����Ȩ�س�ʼ��
weight_input_x=rand(input_num,cell_num)/ab.*MASK;
weight_input_h=rand(cell_num,cell_num)/ab.*MASK;
weight_inputgate_x=rand(input_num,cell_num)/ab.*MASK;
weight_inputgate_c=rand(cell_num,cell_num)/ab.*MASK;
weight_inputgate_c=weight_inputgate_c.*MASK';
weight_forgetgate_x=rand(input_num,cell_num)/ab.*MASK;
weight_forgetgate_c=rand(cell_num,cell_num)/ab.*MASK;
weight_forgetgate_c=weight_forgetgate_c.*MASK';
weight_outputgate_x=rand(input_num,cell_num)/ab.*MASK;
weight_outputgate_c=rand(cell_num,cell_num)/ab.*MASK;
weight_outputgate_c=weight_outputgate_c.*MASK';
% hidden_outputȨ��
weight_preh_h=rand(cell_num,output_num).*MASK';
% ����״̬��ʼ��
h_state=rand(output_num,data_num);
cell_state=rand(cell_num,data_num);
TrainSamInCurrent=[];
cell_numHistory=[];
%--------------------Ϊ������֮ǰ��LSTM�����ͻ�����½��в��ֱ����ĳ�ʼ����
gate=[];
input_gate_input=[];
forget_gate_input=[];
output_gate_input=[];
input_gate=[];
forget_gate=[];
output_gate=[];
pre_h_state=[];
%-----------------------------------------------------------------------
%% ����ѵ��ѧϰ
% for iter=1:60%ѵ������
tic
    for iter=1:data_num  %����һ��һ������ TrainSamNum
    iter
    %     epoch=1;
    TrainSamInEvery=train_data(:,iter);
    TrainSamInCurrent=[TrainSamInCurrent TrainSamInEvery];
    

    %% ��ȡѵ������TrainSamIn��TrainSamNum
    [CurrentIndim,CurrentNum]=size(TrainSamInCurrent);
    
     yita = 0.05;
%    yita=5/(100+0.2*sqrt(iter)); %����Ӧѧϰ��
%     for n=1:numbatches%����minibatch��ʽ����ѵ��
%         train_data=batchdata(:,:,n);
%         train_label=batchdata_target(:,:,n);
    for m=1:CurrentNum
        %ǰ������
        if(m==1)
            gate=tanh(TrainSamInCurrent(:,m)'*weight_input_x.*MASK+bias_gate.*MASK);
            input_gate_input=TrainSamInCurrent(:,m)'*weight_inputgate_x.*MASK+bias_input_gate.*MASK;
            output_gate_input=TrainSamInCurrent(:,m)'*weight_outputgate_x.*MASK+bias_output_gate.*MASK;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            forget_gate=zeros(1,cell_num);
            forget_gate_input=zeros(1,cell_num);
            cell_state(:,m)=(input_gate.*gate)';
        else
            gate=tanh(TrainSamInCurrent(:,m)'*weight_input_x.*MASK+pre_h_state(:,m-1)'*weight_input_h.*MASK+bias_gate.*MASK);
            input_gate_input=TrainSamInCurrent(:,m)'*weight_inputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_inputgate_c.*MASK.*MASK')+bias_input_gate.*MASK;
            forget_gate_input=TrainSamInCurrent(:,m)'*weight_forgetgate_x.*MASK+pre_h_state(:,m-1)'*(weight_forgetgate_c.*MASK.*MASK')+bias_forget_gate.*MASK;
            output_gate_input=TrainSamInCurrent(:,m)'*weight_outputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_outputgate_c.*MASK.*MASK')+bias_output_gate.*MASK;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';   
        end
        pre_h_state(:,m)=tanh(cell_state(:,m)').*output_gate;
        ym(m)= pre_h_state(:,m)'*(weight_preh_h.*MASK');
%         ym(m)=1/(1+exp(-ym_input(:,m)));
        %������
        
        Error=train_label(m)-ym(m);
        
            [   weight_input_x,...
                weight_input_h,...
                weight_inputgate_x,...
                weight_inputgate_c,...
                weight_forgetgate_x,...
                weight_forgetgate_c,...
                weight_outputgate_x,...
                weight_outputgate_c,...
                weight_preh_h,...
                bias_gate,...
                bias_input_gate,...
                bias_forget_gate,...
                bias_output_gate]=LSTM_updata_weight_bias(cell_num,output_num,m,yita,Error,...
                                                   weight_input_x,...
                                                   weight_input_h,...
                                                   weight_inputgate_x,...
                                                   weight_inputgate_c,...
                                                   weight_forgetgate_x,...
                                                   weight_forgetgate_c,...
                                                   weight_outputgate_x,...
                                                   weight_outputgate_c,...
                                                   weight_preh_h,...
                                                   bias_gate,...
                                                   bias_input_gate,...
                                                   bias_forget_gate,...
                                                   bias_output_gate,...
                                                   cell_state,h_state,...
                                                   input_gate,forget_gate,...
                                                   output_gate,gate,...
                                                   train_data,pre_h_state,...
                                                   input_gate_input,...
                                                   output_gate_input,...
                                                   forget_gate_input);
                

%     end

        
      
    if(dropout>0.) %Dropout
        rand('seed',0)
           weight_inputgate_x =weight_inputgate_x.*(rand(size(weight_inputgate_x))>dropout);
    end

    end
    
    for j=1:size(train_data,2)
test_final=train_label(:,j);

        %ǰ������
        if(j==1)
            gate=tanh(train_data(:,j)'*weight_input_x.*MASK);
            input_gate_input=train_data(:,j)'*weight_inputgate_x.*MASK+bias_input_gate.*MASK;
            output_gate_input=train_data(:,j)'*weight_outputgate_x.*MASK+bias_output_gate.*MASK;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            forget_gate=zeros(1,cell_num);
            forget_gate_input=zeros(1,cell_num);
            cell_state(:,j)=(input_gate.*gate)';
        else
            if iter==201
                aaa=1;
            end
            gate=tanh(train_data(:,j)'*weight_input_x.*MASK+pre_h_state(:,j-1)'*(weight_input_h.*MASK.*MASK')+bias_gate.*MASK);
            input_gate_input=train_data(:,j)'*weight_inputgate_x.*MASK+pre_h_state(:,j-1)'*(weight_inputgate_c.*MASK.*MASK')+bias_input_gate.*MASK;
            forget_gate_input=train_data(:,j)'*weight_forgetgate_x.*MASK+pre_h_state(:,j-1)'*(weight_forgetgate_c.*MASK.*MASK')+bias_forget_gate.*MASK;
            output_gate_input=train_data(:,j)'*weight_outputgate_x.*MASK+pre_h_state(:,j-1)'*(weight_outputgate_c.*MASK.*MASK')+bias_output_gate.*MASK;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            cell_state(:,j)=(input_gate.*gate+cell_state(:,j-1)'.*forget_gate)';   
        end
        pre_h_state(:,j)=tanh(cell_state(:,j)').*output_gate;
        ym_train(j)= pre_h_state(:,j)'*(weight_preh_h.*MASK');

end
             error=train_label-ym_train;

cell_numHistory=[cell_numHistory sum(MASK)];
     % ѵ��RMSE������
    TrainError=error;
    TrainRMSE(iter)=sqrt(sum(TrainError.^2)/data_num);
    
%     ����֯LSTM
    if   mod(iter,step_window)==0  %ÿ10��ִ��һ������֯
        TestSamIn_window = train_data(:,iter);
              
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ����ɸѡ MIV�㷨�ĳ���ʵ�֣����ӻ��߼����Ա�����

% x_increase%�ľ��� x_decreaseΪ����10%�ľ���
    x_increase=TestSamIn_window*1.1;
    x_decrease=TestSamIn_window*0.9;
    cell_state_increase=cell_state*1.1;
    cell_state_decrease=cell_state*0.9;
    pre_h_state_increase=pre_h_state*1.1;
     pre_h_state_decrease=pre_h_state*0.9;
    
    
%% ����ɸѡ MIV�㷨�ĺ���ʵ�֣���ֵ���㣩

if iter==800
    aaa=1;
end
% result_inΪ����10%������ result_deΪ����10%������
result_in=LSTM_H(iter,x_increase,weight_input_x,...
                weight_input_h,...
                weight_inputgate_x,...
                weight_inputgate_c,...
                weight_forgetgate_x,...
                weight_forgetgate_c,...
                weight_outputgate_x,...
                weight_outputgate_c,...
                bias_gate,...
                bias_input_gate,...
                bias_forget_gate,...
                bias_output_gate,...
                cell_state_increase,...
                input_gate,forget_gate,...
                output_gate,...
                pre_h_state_increase,cell_num,weight_preh_h,MASK);
            
result_de=LSTM_H(iter,x_decrease,weight_input_x,...
                weight_input_h,...
                weight_inputgate_x,...
                weight_inputgate_c,...
                weight_forgetgate_x,...
                weight_forgetgate_c,...
                weight_outputgate_x,...
                weight_outputgate_c,...
                bias_gate,...
                bias_input_gate,...
                bias_forget_gate,...
                bias_output_gate,...
                cell_state_decrease,...
                input_gate,forget_gate,...
                output_gate,...
                pre_h_state_decrease,cell_num,weight_preh_h,MASK);
            
                result_normal=LSTM_H(iter,TestSamIn_window,weight_input_x,...
                weight_input_h,...
                weight_inputgate_x,...
                weight_inputgate_c,...
                weight_forgetgate_x,...
                weight_forgetgate_c,...
                weight_outputgate_x,...
                weight_outputgate_c,...
                bias_gate,...
                bias_input_gate,...
                bias_forget_gate,...
                bias_output_gate,...
                cell_state,...
                input_gate,forget_gate,...
                output_gate,...
                pre_h_state,cell_num,weight_preh_h,MASK);

deta_x=result_in-result_de;
deta_x_abs=abs(deta_x);
%% �ҳ�ռ��ǰ85%�ı���
deta_x_sum = sum(deta_x_abs);
[deta_x_sort,index]=sort(deta_x_abs,'descend');
deta_x_result = 0;
i=1 ;
while (deta_x_result/deta_x_sum)<0.9

    deta_x_result = deta_x_result+deta_x_sort(i);
    i=i+1;    
end


MASK0=MASK;
        %�ṹ����֯
        %��֦
        
                beta=0.5;
A=abs(result_normal);
non_zero_num=sum(A~=0);
[B,index_B]=sort(A,'descend');
Threshold_result_normal=B(round((beta)*non_zero_num));
[prune_index0,a]=find(A>Threshold_result_normal);
       
      prune_index=setdiff(index(i:end),prune_index0);

        if i<size(index,1)&&~isempty(prune_index)
%         
            MASK(prune_index)=0;

        end
       
%         %����.
alpha=0.8;
A=abs(weight_preh_h.*MASK');
non_zero_num=sum(A~=0);
B=sort(A,'descend');
Threshold_weight_preh_h=B(ceil((1-alpha)*non_zero_num));
 [grow_index,b]=find(abs(weight_preh_h.*~MASK')>Threshold_weight_preh_h);

  AA=abs(weight_preh_h);
  [grow_index1,b]=find(abs(weight_preh_h)>mean(AA(AA~=0)));
grow_index=union(grow_index,grow_index1);
if size(grow_index,1)>0

    MASK(grow_index)=1;

end

        
grow_index_final=setdiff(find(MASK==1),find(MASK0==1));
if ~isempty(grow_index_final)&&any(pre_h_state(grow_index_final,iter))
    weight_preh_h(grow_index_final)=Error./pre_h_state(grow_index_final,iter);
end

prune_index_final=setdiff(find(MASK==0),find(MASK0==0));
if ~isempty(prune_index_final)&&any(pre_h_state(index_B(1),iter))
    MASK_prune=zeros(1,cell_num);
    MASK_prune(prune_index_final)=1;
    weight_preh_h(index_B(1))=weight_preh_h(index_B(1))+pre_h_state(:,iter)'*(weight_preh_h.*MASK_prune')./pre_h_state(index_B(1),iter);

end


    end
    
    
    end
    toc

    %% ѵ��Ч���׶�
%���ݼ���
for j=1:size(train_data,2)
test_final=train_data(:,j);

%ǰ��
m=data_num+j;
gate=tanh(test_final'*weight_input_x.*MASK+pre_h_state(:,m-1)'*(weight_input_h.*MASK.*MASK')+bias_gate.*MASK);
input_gate_input=test_final'*weight_inputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_inputgate_c.*MASK.*MASK')+bias_input_gate.*MASK;
forget_gate_input=test_final'*weight_forgetgate_x.*MASK+pre_h_state(:,m-1)'*(weight_forgetgate_c.*MASK.*MASK')+bias_forget_gate.*MASK;
output_gate_input=test_final'*weight_outputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_outputgate_c.*MASK.*MASK')+bias_output_gate.*MASK;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
        pre_h_state(:,m)=tanh(cell_state(:,m)').*output_gate;
        ym= pre_h_state(:,m)'*(weight_preh_h.*MASK');

SOLSTM_train_sim(:,j)=ym;
end


ym_train_SOLSTM=mapminmax('reverse',SOLSTM_train_sim,outputps);%����Ԥ������
figure
plot(output_train,'-bo');
grid on
hold on
plot(ym_train_SOLSTM,'-r*');
legend('Ԥ������','ʵ������')
title('LSTM������ع�Ԥ��')
xlabel('������')
ylabel('LSTMԤ��ѵ�����')

figure
plot(TrainRMSE)
xlabel('��������')
ylabel('ѵ�����')
title('LSTMѵ���������')
error=output_train-ym_train_SOLSTM;
rmse_train=sqrt(mean(((error).^2)))

%%%%%%
Pn_train=XTrain';
Pn_test=XTest';

Tn_train=output_train;
Tn_test=output_test;

%% ���Խ׶�
%���ݼ���
for j=1:size(P_test,2)
test_final=P_test(:,j);

%ǰ��
m=data_num+j;
gate=tanh(test_final'*weight_input_x.*MASK+pre_h_state(:,m-1)'*(weight_input_h.*MASK.*MASK')+bias_gate.*MASK);
input_gate_input=test_final'*weight_inputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_inputgate_c.*MASK.*MASK')+bias_input_gate.*MASK;
forget_gate_input=test_final'*weight_forgetgate_x.*MASK+pre_h_state(:,m-1)'*(weight_forgetgate_c.*MASK.*MASK')+bias_forget_gate.*MASK;
output_gate_input=test_final'*weight_outputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_outputgate_c.*MASK.*MASK')+bias_output_gate.*MASK;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
        pre_h_state(:,m)=tanh(cell_state(:,m)').*output_gate;
        ym= pre_h_state(:,m)'*(weight_preh_h.*MASK');
%         ym=1/(1+exp(-ym_input));

sim(:,j)=ym;
end

%����һ��
ym_test_SOLSTM=mapminmax('reverse',sim,outputps);%����Ԥ������
test=T_test';%ʵ������

%%
%%

%��Ԫ�仯��
figure;
plot(cell_numHistory,'K-','LineWidth',2)
axis tight;
xlabel('Training samples','fontsize',10.5,'fontname','TimesNewRoman')
ylabel('Number of neurons','fontsize',10.5,'fontname','TimesNewRoman')

ylim([min(cell_numHistory)-1 max(cell_numHistory)+1])

%%
%%
figure
plot(test,'r*');
% grid on
hold on
plot(ym_test_SOLSTM,'-bo');
axis tight;
h1=legend('Real values','SOLSTM output');
set(h1,'Orientation','horizon','FontSize',10)
legend('boxoff')
xlabel('Time steps')
ylabel('Outputs')
error_SOLSTM=test-ym_test_SOLSTM';
ylim([0 0.6])

% ���
figure
plot(error_SOLSTM)
axis tight;
ylabel('Prediction error')
xlabel('Time steps')

Rmse_SOLSTM = sqrt(mean(((error_SOLSTM).^2)))

%Mapeƽ���ٷֱ����
Mape_SOLSTM = mean(abs((error_SOLSTM')./output_test))

%MAEƽ���������
Mae_SOLSTM = mean(abs(error_SOLSTM))

aaa =1;