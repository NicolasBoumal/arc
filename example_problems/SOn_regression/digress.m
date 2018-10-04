function optim_problem = digress(regression_problem)
% DIGRESS algorithm for discrete regression on SO(n).
%
% See the paper:
%
% Interpolation and Regression of Rotation Matrices,
% Nicolas Boumal, 2013
% In: Nielsen F., Barbaresco F. (eds) Geometric Science of Information.
% Lecture Notes in Computer Science, vol 8085. Springer, Berlin, Heidelberg
% https://link.springer.com/chapter/10.1007/978-3-642-40020-9_37
% 
% Code was adapted from the original in October 2017.
%
% Input:
%
%   regression_problem
%       Structure describing the regression problem. Fields are as defined
%       in the example script main.m.
%
% Output is for use with Manopt, freely available at http://www.manopt.org.
%
% Nicolas Boumal, Oct. 2017.

    n = regression_problem.n;
    Nd = regression_problem.Nd;
    
    % Obtain Manopt description of the manifold SO(n)^Nd.
    manifold = rotationsfactory(n, Nd);

    % Define the cost function, at X.
    function [E, store] = mycost(X, store)
        if ~isfield(store, 'E')
            [store, E] = cost(store, regression_problem, X);
            store.E = E;
        end
        E = store.E;
    end

    % Define the gradient of the cost, at X.
    function [gradE, store] = mygrad(X, store)
        if ~isfield(store, 'gradE')
            [store, ~, gradE] = cost(store, regression_problem, X);
            store.gradE = manifold.egrad2rgrad(X, gradE);
        end
        gradE = store.gradE;
    end

    % Define the Hessian of the cost, at X, along XO.
    function [hessE, store] = myhess(X, O, store)
        [store, ~, gradE, hessE] = cost(store, regression_problem, X, O);
        hessE = manifold.ehess2rhess(X, gradE, hessE, O);
        store = incrementcounter(store, 'hesscalls');
    end
    
    % Combine the manifold, the cost and its derivatives into a problem
    % structure for Manopt.
    optim_problem.M = manifold;
    optim_problem.cost = @mycost;
    optim_problem.grad = @mygrad;
    optim_problem.hess = @myhess;
    
end
