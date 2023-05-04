function [fx] = Rosenbrock100(x)
% Funci�n de Rosenbrock de dimensi�n 100. Es la generalizaci�n
% de la funci�n de Rosenbrock de dimensi�n 2.
% f(x) = sum_{k=1}^{50} [ (x(2*k) - x(2*k-1)^2)^2 + (1 -x(2*k-1))^2]
% El m�nimo de la funci�n es le punto x* = [1, 1, . . ., 1]';
% In.
% x.- vector columna de dimensi�n 100.
% Out
%fx.- n�mero real.
%
% An�lisis Aplicado
% ITAM
%  6 de noviembre de 2015

n = length(x);

if (n==100)
  fx = 0;
  for k = 1:50
      fx = fx + (x(2*k) - x(2*k-1)^2)^2;
      fx = fx +(1 -x(2*k-1))^2;
  end  
else
    disp('La dimension no es correcta')
    fx =[];
end