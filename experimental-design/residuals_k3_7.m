function SSE = residuals_k3_7(k2,tVec,conc,constant_params,k3_range,d1,d2,d3,d4,d5,d6,d7) 

    IC= [0 1 0 10];
    params_passed = [constant_params(1:3) k2 k3_range constant_params(4:5)]; 
    sol = ode45(@(t,c) oDE(t,c,params_passed) ,[0 length(tVec)],IC); 
    y = deval(sol,tVec,4);
    sse1 = sum((y(d1)-conc(d1)).^2); 
    sse2 = sum((y(d2)-conc(d2)).^2); 
    sse3 = sum((y(d3)-conc(d3)).^2); 
    sse4= sum((y(d4)-conc(d4)).^2); 
    sse5=sum((y(d5)-conc(d5)).^2); 
    sse6=sum((y(d6)-conc(d6)).^2);
    sse7=sum((y(d7)-conc(d7)).^2); 
    

    SSE= sse1+sse2+sse3+sse4+sse5+sse6+sse7;

end