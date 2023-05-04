fname = 'extendedPenalty';
x = (1:1000)';

% Método de descenso de Newton
[xfN, jN] = desnewton(fname, x);


gN = gradiente(fname,xfN);
ngN= norm(gN);




% Método BFGS

[xfB, jB ] = bfgs(fname, x)


gB = gradiente(fname,xfB);
ngB = norm(gB);


% Método DFP

[xfD,MGD,jD] = dfp(fname, x);


gD = gradiente(fname,xfD);
ngD = norm(gD);


semilogy([1:length(MGD)], MGD, 'rd', [1:length(MGD)], MGD, 'b', 'Linewidth', 3)
title('Convergencia de DFP', 'Fontsize',20)
xlabel('Número de iteraciones', 'Fontsize',20);
ylabel('||g||', 'Fontsize',20);

%semilogy([1:length(ngB)], ngB, 'rd', [1:length(ngB)], ngB, 'b', 'Linewidth', 3)
%title('Convergencia de ngB', 'Fontsize',20)
%xlabel('Número de iteraciones', 'Fontsize',20);
%ylabel('||g||', 'Fontsize',20);

%semilogy([1:length(ngN)], ngN, 'rd', [1:length(ngN)], ngN, 'b', 'Linewidth', 3)
%title('Convergencia de ngN', 'Fontsize',20)
%xlabel('Número de iteraciones', 'Fontsize',20);
%ylabel('||g||', 'Fontsize',20);


sz = [4 4];
varTypes = ["string","int16","double", "double"];
varNames = ["Método","Núm. iter", "||∇f(x*)||2","cputime"];
T = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

T(1,varNames) = {"Método de Newton",jN,ngN,.5};
T(2,varNames) ={"Método BFGS", jB,ngB,.5 };
T(3,varNames) ={"Método DFP", jD,ngD,.5 }


%falta poner fminunc

