%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   code to produce scatter plot of the different combinations of 
%%%      of data collection days
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear vars; 
load("k3_profiles_7days.mat","doublecheckdays_k3");
numPoints = 300;
x = rand(1, numPoints) - 0.5;
y = rand(size(x)) - 0.5;
% Sizes = distance from origin
sizes = 200 * sqrt(x.^2 + y.^ 2);
% Colors according to the jet colormap and distance from origin.
indexes = round(rescale(sizes, 1, numPoints));
colors = jet(numPoints);	% Initialize
colors = colors(indexes,:); % Colored according to distance.
% Do the plot
scatter(x, y, sizes, colors, 'filled');
grid on;


A=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1; 
    2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2; 
    3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3; 
    4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4; 
    5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5; 
    6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6; 
    7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7];
Avec = reshape(A',1,[]); %transpose matrix

B = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
Bvec = reshape(B',1,[]); %transpose matrix

count=zeros(7,20); %vec to count the frequency of each day

%for loop to count the frequence of each day to populate count vec
for i=1:7
    y = doublecheckdays_k3(:,i);
    for j=1:52 
        for k=1:20

            if y(j)==k
                count(i,k) =count(i,k)+1;
            end
        
        end

    end 
        
     
end

%for loop to shift any index containing a 0 to avoid error in producing
%graph
for i=1:7
    for j=1:20
        if count(i,j)==0
            count(i,j) = count(i,j)+0.00000000001;
        end
    end
end

count_vec = reshape(count',[],1);
count_vec=10*count_vec; %scaling of the circles

figure;
scatter(Avec,Bvec,count_vec,"filled")
xlim([0,8])
ylim([0,21])
xlabel('Experimental Time Points')
ylabel('Day Number')
title('Optimal Number of Days')
saveas(gcf,'Optimal Number of Days','jpg');