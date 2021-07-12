%% F1SCORE Optimal Tb Ts to get precision recall
clc
close all;
clear all;
[ind_x ind_y Tb Ts TP FP TN FN] = textread('Recording.txt','%d %d %f %f %d %d %d %d');
for i=1:size(TP,1)
    precision(i,1)=TP(i,1)/(TP(i,1)+FP(i,1));
    recall(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    true_positive_rate(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    false_positive_rate(i,1)=FP(i,1)/(FP(i,1)+TN(i,1));
    accuracy(i,1)=1-(FP(i,1)+FN(i,1))/(TP(i,1)+FP(i,1)+FN(i,1)+TN(i,1));
end
for i=1:size(precision,1)
    F1SCORE(i,1)=(2*precision(i,1)*recall(i,1))/(precision(i,1)+recall(i,1));
end
[max index]=max(F1SCORE);% 其中一組解
fprintf('Optimal Tb=%d Ts=%d　precision=%f recall=%f \n',Tb(index,1),Ts(index,1),precision(index,1),recall(index,1))

%% draw precision
clc
close all;
clear all;
[ind_x ind_y Tb Ts TP FP TN FN] = textread('Recording1.txt','%d %d %f %f %d %d %d %d');
for i=1:size(TP,1)
    precision(i,1)=TP(i,1)/(TP(i,1)+FP(i,1));
    recall(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    true_positive_rate(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    false_positive_rate(i,1)=FP(i,1)/(FP(i,1)+TN(i,1));
    accuracy(i,1)=1-(FP(i,1)+FN(i,1))/(TP(i,1)+FP(i,1)+FN(i,1)+TN(i,1));
end
[X,Y]=meshgrid(0:0.01:1,0:0.01:1);
k=1;
for i=1:1:size(X,1)
   for j=1:1:i
       if isnan(precision(k,1));
           Z(i,j)=-1;
       else
           Z(i,j)=precision(k,1);
       end
       k=k+1;
   end 
end
figure(1)
surf(X,Y,Z)
title('Precision')
xlabel('Tb')
ylabel('Ts')
zlabel('precision')

%% draw accuracy
clc
close all;
clear all;
[ind_x ind_y Tb Ts TP FP TN FN] = textread('Recording.txt','%d %d %f %f %d %d %d %d');
for i=1:size(TP,1)
    precision(i,1)=TP(i,1)/(TP(i,1)+FP(i,1));
    recall(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    true_positive_rate(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    false_positive_rate(i,1)=FP(i,1)/(FP(i,1)+TN(i,1));
    accuracy(i,1)=1-(FP(i,1)+FN(i,1))/(TP(i,1)+FP(i,1)+FN(i,1)+TN(i,1));
end
[X,Y]=meshgrid(0:0.01:1,0:0.01:1);
k=1;
for i=1:1:size(X,1)
   for j=1:1:i
       if isnan(precision(k,1));
           Z(i,j)=-1;
       else
           Z(i,j)=precision(k,1);
       end
       k=k+1;
   end 
end
figure(1)
surf(X,Y,Z)
title('accuracy')
xlabel('Tb')
ylabel('Ts')
zlabel('accuracy')

%% draw PR curve
clc
close all;
clear all;
[ind_x ind_y Tb Ts TP FP TN FN] = textread('Recording.txt','%d %d %f %f %d %d %d %d');
for i=1:size(TP,1)
    precision(i,1)=TP(i,1)/(TP(i,1)+FP(i,1));
    recall(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    true_positive_rate(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    false_positive_rate(i,1)=FP(i,1)/(FP(i,1)+TN(i,1));
    accuracy(i,1)=1-(FP(i,1)+FN(i,1))/(TP(i,1)+FP(i,1)+FN(i,1)+TN(i,1));
end
figure;
PR=[recall precision];
PR=sortrows(PR,1)
plot(PR(:,1),PR(:,2),'r-','linewidth',2);
xlabel('Recall')
ylabel('Precision')
axis square
%% draw ROC curve
clc
close all;
clear all;
[ind_x ind_y Tb Ts TP FP TN FN] = textread('Recording.txt','%d %d %f %f %d %d %d %d');
for i=1:size(TP,1)
    precision(i,1)=TP(i,1)/(TP(i,1)+FP(i,1));
    recall(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    true_positive_rate(i,1)=TP(i,1)/(TP(i,1)+FN(i,1));
    false_positive_rate(i,1)=FP(i,1)/(FP(i,1)+TN(i,1));
    accuracy(i,1)=1-(FP(i,1)+FN(i,1))/(TP(i,1)+FP(i,1)+FN(i,1)+TN(i,1));
end
figure;
ROC=[true_positive_rate false_positive_rate];
ROC=sortrows(ROC,1)
plot(ROC(:,1),ROC(:,2),'r-','linewidth',2);
xlabel('TPR')
ylabel('FPR')
axis square