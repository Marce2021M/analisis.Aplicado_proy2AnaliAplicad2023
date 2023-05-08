function [ xf, MG,j ] = bfgs(fname, x)
% 

tol = 1e-6;  % tolerancia a la norma del gradiente
jmax = 100;   % número máximo de iteraciones
c1 = 0.0001;     % parámetro para np dar pasos grandes
kmax = 10;     % número  máximo de pasos hacia atrás

g = gradiente(fname,x);  fx = feval(fname,x);
MG = norm(g);
n = length(x);
A = eye(n);
j = 0;
while (norm(g) > tol && j < jmax )
    %% Regularization of the hessian

    % Calculate the eigenvalues of the Hessian matrix. 
    % We need to make sure that all eigenvalues are positive, 
    % otherwise we adjust the Hessian matrix to make all eigenvalues positive.

    vhessian = eig(A);           % eigenvalues of the hessian
    vminHessian = min(vhessian);       % minimum eigenvalue of the hessian
    
    % Ensure that the hessian is positive definite by adding a small positive value to its diagonal elements.
    if (vminHessian <= 0)
        A = A + (abs(vminHessian) + 0.5) * eye(n);
    end
    %%
    p = pcg(A, -g, 1e-06,100);    % dirección de descenso
    
    % búsqueda de línea
    alfa = 1.0;
    xp = x + alfa*p;
    fxp = feval(fname,xp);
    k = 0;
    while( fxp > fx + alfa*( c1*g'*p ) && k < kmax)
        alfa = alfa/2;
        xp = x + alfa*p;
        fxp = feval(fname,xp);  
        k = k+1;
    end
    %------------------------------------
    if (k ==kmax)
        alfa = 1.0;
    end
    xp = x + alfa*p;
    gxp = gradiente(fname,xp);
    s = xp -x;  y = gxp-g;
    
    u = A*s;
    A = A + ( (y*y')/(y'*s) ) - ((u*u')/(s'*A*s) ) ;
        
    x = xp;
    g = gxp;
    MG = [ MG ; norm(g)];
    j = j+1;
    %fprintf('%2.0f %2.8f \n ',j,norm(g)  );
end

xf = x;

end

