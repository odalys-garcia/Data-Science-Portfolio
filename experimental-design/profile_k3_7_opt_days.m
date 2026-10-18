% % % % % % % % % % % % FOR K3 % % % % % % % % % % % % % % %
%% Purpose  of this program is to determine what singular day 
 %% produces the best profile for k3 %%
clearvars; close all;
rng(1);
load("one_dose_1drug.mat", "peripheral_conc","central_conc");
load one_dose_1drug.mat;
load("simulated_fitting_multi.mat","params_opt","minSSE");
load("count_k3_7.mat","id");
load("/Users/odalys/Documents/MATLAB/MAT 493/output_k3profiles_7days_3281407/k3_profiles_7days.mat", ...
    "time_days");


cl = clock; clN = 0;
for ii = 2:5
    clN = floor(100*clN + cl(ii));
end
path = ['output_k3profiles_7days_opt_days' , num2str(clN)];
mkdir(path)


conc = central_conc;
LB = [0 0 0 0 0 0 0]; 
UB= 3*[1 .05 0.1 0.25 0.1 1 1];
UB(4) =1;
UB(5)=1;
params_int = (LB +UB)/2;
k_onA= 1; k_onA_range = linspace(0.75,1.65,10); 
k_offA =0.05; k_offA_range = linspace(0.05,0.12,7); 
k1 = 0.1; k1_range = linspace(0,.1,5);  
k2= params_int(4); k2_range= linspace(0.1,.4,10);
k3= params_int(5); k3_range = logspace(-3,0,20);
j= 1; j_range = linspace(0,1,10);
d=1; d_range = linspace(0,1,10);



constant_params = [k_onA k_offA k1 j d];

ICs_initial= [0 1 0 10];


options = optimoptions(@fmincon,'OptimalityTolerance',1e-12,'StepTolerance',1e-15,...
    'display','notify');
A = []; b= [];   % Linear inequality constraints of form Ax <= b
Aeq = []; beq = []; % Linear equality constrains of form Aeq*x = beq
numParam = 1; % how many initial conditions you want 
n_skip = 1000; n_leap = 0; % Parameters needed by sobolset - not very important
numICs=50;
threshold_val = chi2inv(0.9,numParam)/2 + minSSE;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FITTING OVER THE SPECIFIED PARAMETER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% Open parallel pool
npools = 4; % max 4 at home, 30? on cluster
parpool('local', npools); % Open distributed processing 
poolobj = gcp('nocreate'); % If no pool, don’t create
if isempty(poolobj)
 	poolsize = 0;
else
	poolsize = poolobj.NumWorkers;
end


parfor m =1:length(opt_days)
    r1 = opt_days(m,1);
    r2= opt_days(m,2);
    r3=opt_days(m,3);
    r4=opt_days(m,4);
    r5 = opt_days(m,5);
    r6 = opt_days(m,6);
    r7 = opt_days(m,7);
    time_days(m,:) = [tVec(r1) tVec(r2) tVec(r3) tVec(r4) tVec(r5) tVec(r6) tVec(r7)];
    bestp_k3_temp = zeros(1,length(k3_range));
    k3_SSE_temp = zeros(1,length(k3_range));
    
    for i=1:length(k3_range)
        fprintf('For m = %d: up to k3_range = %f\n',m,k3_range(i));
        k3_pt = k3_range(i);
        dy = @(z)residuals_k3_7(z,tVec,conc,constant_params,k3_pt,r1,r2,r3,r4,r5,r6,r7);
        [bestp_k3_temp(i),k3_SSE_temp(i)] = fmincon(dy,k2,A,b,Aeq,beq,LB(4),UB(4),[],options);
    end

    bestp_k3(m,:) = bestp_k3_temp;
    k3_SSE(m,:) = k3_SSE_temp;


end

fout = [path '/k3_profiles_7days.mat']; 
save(fout,'time_days','bestp_k3','k3_SSE','threshold_val','opt_days'); 



%% Close parallel pool
delete(gcp('nocreate'));


