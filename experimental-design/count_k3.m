% % % % % % % % % % % % FOR K3 % % % % % % % % % % % % % % %
%% purpose of this code is to count the number of combinations in which
 %% one/both ends of the graph are above the threshold%%
clearvars; close all;
rng(1);
load("/Users/odalys/Documents/MATLAB/MAT 493/output_k3profiles_7days_3281407/k3_profiles_7days.mat", ...
    "time_days","bestp_k3","k3_SSE","threshold_val");

%ck after counter [a,b] =max(k3_SSE,[],1)


v = 2:1:21;
t_sample = nchoosek(v,7);
q = length(t_sample)/10;
counter = zeros(q,1);
%increment index if one/both ends are above the threshold
for i= 1:length(k3_SSE)

    p1 = k3_SSE(i,1);
    p2 = k3_SSE(i,end);

    if(p1 > threshold_val)

        counter(i,:) = counter(i) +1;

    end 

    if(p2>threshold_val)

        counter(i,:) = counter(i)+1;
    end

end

k=0;
id=zeros(7,1);
%prints the index of each combination where both ends are above the
%threshold
for i=1:length(counter) 
    if(counter(i,:)== 2)
        k=k+1;
        id(k,:)=i;
    end
end
save("count_k3_7.mat","counter","id")
% max value of the left endpt reaches 2.1327 at time combination 186
% days are: 1,2,3,9,10,12,13,14,15,16,19,20
