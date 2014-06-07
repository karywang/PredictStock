tic;
close all;
clear;
clc;
format compact;

load train2.mat;
[m,n] = size(train2);
ts1 = train2(:,1);
tsx1 = train2(:,2:end);

load test2.mat;
[m1,n1] = size(test2);
ts2 = test2(:,1);
tsx2 = test2(:,2:end);

[tsx1,tsx2]= scaling(tsx1,tsx2,1,2);

A = mean(tsx1);
B = repmat(A,m,1);
X = tsx1 - B;
C = X *X';
[v1,d1]=eig(C); 
[dummy,order] = sort(diag(-d1));
disp(dummy);
v1 = v1(:,order);%将特征向量按照特征值大小进行降序排列，每一列是一个特征向量
dsum = sum(dummy);
dsum_extract = 0;
p = 0;
while( dsum_extract/dsum < 0.95)
    p = p + 1;
    dsum_extract = sum(dummy(1:p));
end
i = 1;
% (训练阶段)计算特征脸形成的坐标系
while (i<=p)
    base(:,i) = X' *v1(:,i)/norm(v1(:,i));;   % base是N×p阶矩阵
    i = i + 1
end
Y = X*base;

% 
% [bestmse,bestc,bestg] = SVMcgForRegress(ts1,Y,-10,10,-10,10);
% % 打印粗略选择结果
% disp('打印粗略选择结果');
% str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
% disp(str);
% 
% % 根据粗略选择的结果图再进行精细选择: 
% [bestmse,bestc,bestg] = SVMcgForRegress(ts1,Y,-4,4,-4,4);
% % 打印精细选择结果
% disp('打印精细选择结果');
% str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
% disp(str);

%bestc = 16;
%bestg = 0.32988;

bestc = 16;
bestg = 0.0008;

%% 利用回归预测分析最佳的参数进行SVM网络训练
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
model = svmtrain(ts1,Y,cmd);
%% SVM网络回归预测
[predict,mse] = svmpredict(ts1,Y,model);


B1 = repmat(A,m1,1);
X1 = tsx2 - B1;
Y1 = X1 * base;

[predict1,mse1] = svmpredict(ts2,Y1,model);
