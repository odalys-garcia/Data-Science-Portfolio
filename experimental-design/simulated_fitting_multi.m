%% the purpose of this code is to produce simulated data needed to find the optimal number
%% of days 
clearvars; close all;
rng(1);
load("one_dose_1drug.mat", "peripheral_conc","central_conc");
load one_dose_1drug.mat;

tVec = [0:20];
time= tVec(1):1:tVec(end);
conc = central_conc;
ydata = [time conc];
LB = [0 0 0 0 0 0 0]; 
%UB = [30 50 50 20 50 50 100];  
UB= 3*[1 .05 0.1 0.25 0.1 1 1];
% [0.05 0.3 0.32 8 0.02 0.05 2];  %%play with bounds & fitting and how it works with dose response
% UB = [0.3 5 5 20 5 5 10];  
params_int = (LB +UB)/2;
k_onA= params_int(1); k_onA_range = linspace(0,0.3,3);
k_offA = params_int(2); k_offA_range = linspace(0.1,0.9,3);
k1 = params_int(3); k1_range = linspace(0,1,3); 
k2= params_int(4); k2_range= linspace(0,1,3);
k3= params_int(5); k3_range = logspace(-4,0,4);
j= params_int(6); j_range = linspace(0,1,3);
d=params_int(7);
ICs_initial= [0 1 0 10];
% A = []; b= [];   % Linear inequality constraints of form Ax <= b
% Aeq = []; beq = []; % Linear equality constrains of form Aeq*x = beq
options = optimoptions(@fmincon,'OptimalityTolerance',1e-12,'StepTolerance',1e-15);

numParam = 7; % how many initial conditions you want 
n_skip = 1000; n_leap = 0; % Parameters needed by sobolset - not very important
numICs=50;
uniform_sobol = sobolset(numParam,'Skip',n_skip,'Leap',n_leap);
uniform_sobol = net(uniform_sobol,numICs); % note: all sampled values in [0,1]
A = []; b= [];   % Linear inequality constraints of form Ax <= b
Aeq = []; beq = []; % Linear equality constrains of form Aeq*x = beq

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FITTING OVER THE SPECIFIED PARAMETER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:numICs
    initial_conditions(i,:) = (UB-LB).*uniform_sobol(i,:)+LB;
    dy = @(z)residuals(z,time,conc);
    [bestparams(i,:),p_SSE(i,:)] = fmincon(dy,initial_conditions(i,:),A,b,Aeq,beq,LB,UB,[],options);
end

[minSSE, PIndex]=min(p_SSE);
x=bestparams(PIndex,:);
params_opt=[x(1:7)];
solk = ode45(@(t,c) oDE(t,c,params_opt) ,[0, length(time)],ICs_initial);
SSE=residuals(params_opt,time,conc);
save("simulated_fitting_multi.mat","params_opt","minSSE")


figure(1)
plot(time,conc, 'o'); hold on; 
plot(solk.x,solk.y(4,:));
xlabel('Time'); ylabel('Central Concentration');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                     PKPD-ODE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dcdt = oDE(t,c,params_passed)
k_onA =params_passed(1); 
k_offA = params_passed(2); 
k1= params_passed(3);
k2= params_passed(4);
k3 = params_passed(5);
j=params_passed(6);
d=params_passed(7);
dcdt = zeros(4,1);

dcdt(1) = d*k2*c(4)-k_onA*c(1)*c(2) + k_offA*c(3)-k3*c(1)-k1*c(1); %C_A/peripheral
dcdt(2) = -k_onA*c(1)*c(2) + k_offA*c(3); %R
dcdt(3) = k_onA*c(1)*c(2) - k_offA*c(3); %RC_A
dcdt(4) = -k1*c(4)-k2*c(4) +j*k3*c(1); %central compartment
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RESIDUALS FOR RARAMETER

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SSE = residuals(params_int,time,conc) %residuals for fitting k_onA

    IC= [0 1 0 10];
    params_passed = [params_int(1:7)];
    sol = ode45(@(t,c) oDE(t,c,params_passed) ,[0, length(time)],IC);
    y = deval(sol, time,4);
    SSE = sum((y-conc).^2); 

end




