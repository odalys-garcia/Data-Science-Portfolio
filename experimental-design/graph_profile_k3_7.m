clearvars; close all;
rng(1);
load("/Users/odalys/Documents/MATLAB/MAT 493/output_k3profiles_7days_opt_days3281434/k3_profiles_7days.mat", ...
    "time_days","bestp_k3","k3_SSE","threshold_val","opt_days");
load("count_k3_7.mat","id");
load("simulated_fitting_multi.mat","params_opt","minSSE");
load("k3_profile.mat","smallSSE_k3");

cl = clock; clN = 0;
for ii = 2:5
    clN = floor(100*clN + cl(ii));
end
path = ['output_k3profiles_7days_profile' , num2str(clN)];
mkdir(path)

k3_range = logspace(-3,0,20); %remember to change range to match parallel code
tVec = [0:20];

r=length(id);
opt_days = zeros(r,7);

for i=1:7752
    for j=1:length(id)

        if(i == j)

            opt_days(j,1) = time_days(id(i),1);
            opt_days(j,2) = time_days(id(i),2);
            opt_days(j,3) = time_days(id(i),3);
            opt_days(j,4) = time_days(id(i),4);
            opt_days(j,5) = time_days(id(i),5);
            opt_days(j,6) = time_days(id(i),6);
            opt_days(j,7) = time_days(id(i),7);

        end
    end
end


cmap=parula(length(id));
Ncol = ceil((length(id)+2)/12); % at most 12 per column
figure;  hold on; 
 for w = 1:length(id)
       % plot profile corresponding to each sample
         h1 = semilogx(k3_range,k3_SSE(w,:),'-','LineWidth',2,...
                'Color', cmap(w,:),'DisplayName',...
      sprintf('%d&%dd',opt_days(w,1),opt_days(w,2),opt_days(w,3),opt_days(w,4),opt_days(w,5),opt_days(w,6),opt_days(w,7))); 
               h1.Color(4) = 0.5; 



       % profile if have all data
        h2 = semilogx(k3_range,smallSSE_k3,'-k','LineWidth',3,...
            'DisplayName','All time points');
      % threshold 

        h3 = yline(threshold_val,'--k','LineWidth',1,...
            'DisplayName','Confidence Interval');

 end
xlabel("K3 values"); ylabel("SSE");
title("K3 Profile Likelihood Curve")
saveas(gcf,'K3 Profile Likelihood Curve','jpg');