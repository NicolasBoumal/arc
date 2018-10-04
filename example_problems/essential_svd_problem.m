function problem = dominant_invariant_subspace_problem(N)
    
    % Make data for the test
    if ~exist('N', 'var') || isempty(N)
        N = 2; % Number of matrices to process in parallel.
    end  
    A = multiprod(multiprod(randrot(3, N), essential_hat3([0; 0; 1])), randrot(3, N));
    
    % The essential manifold
    M = essentialfactory(N);
    problem.M = M;

    % Function handles of the essential matrix E and Euclidean gradient and Hessian
    costE  = @(E) 0.5*sum(multisqnorm(E-A));
    egradE = @(E) E - A;
    ehessE = @(E, U) U;

    
    % Manopt descriptions
    problem.cost = @cost;
    function val = cost(X)
        val = essential_costE2cost(X, costE); % Cost
    end
    
    problem.egrad = @egrad;
    function G = egrad(X)
        G = essential_egradE2egrad(X, egradE); % Converts gradient in E to X.
    end
    
    problem.ehess = @ehess;
    function [H, store] = ehess(X, S, store)
        H = essential_ehessE2ehess(X, egradE, ehessE, S); % Converts Hessian in E to X.
        store = incrementcounter(store, 'hesscalls');
    end

    problem.name = sprintf('Essential SVD, Essential(%d)', N);

end
