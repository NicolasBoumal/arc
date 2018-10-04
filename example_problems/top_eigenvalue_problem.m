function problem = top_eigenvalue(A, n)
    
    % Generate the problem data.
    if ~exist('n', 'var') || isempty(n)
        if ~exist('A', 'var') || isempty(A)
            n = 128;
        else
            n = size(A, 1);
        end
    end
    if ~exist('A', 'var') || isempty(A)
        A = randn(n);
        A = (A+A')/2;
    end
    
    % Create the problem structure.
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % Define the problem cost function, its gradient and Hessian.
    function store = prepare(x, store)
        if ~isfield(store, 'Ax')
            store.Ax = A*x;
        end
    end
    problem.cost = @cost;
    function [f, store] = cost(x, store)
        store = prepare(x, store);
        Ax = store.Ax;
        f = -x'*Ax;
    end
    problem.egrad = @egrad;
    function [G, store] = egrad(x, store)
        store = prepare(x, store);
        Ax = store.Ax;
        G = -2*Ax;
    end
    problem.ehess = @ehess;
    function [H, store] = ehess(x, xdot, store) %#ok<INUSL>
        H = -2*A*xdot;
        store = incrementcounter(store, 'hesscalls');
    end

    problem.name = sprintf('Top Eigenvalue, S(%d)', n);

end
