function problem = truncated_svd_problem(A, m, n, p)
    
    % Generate some random data to test the function if none is given.
    if ~exist('A', 'var') || isempty(A)
        if ~exist('m', 'var') || isempty(m)
            m = 42;
        end
        if ~exist('n', 'var') || isempty(n)
            n = 60;
        end
        A = randn(m, n);
    end
    if ~exist('p', 'var') || isempty(p)
        p = 5;
    end
    
    % Retrieve the size of the problem and make sure the requested
    % approximation rank is at most the maximum possible rank.
    [m, n] = size(A);
    assert(p <= min(m, n), 'p must be smaller than the smallest dimension of A.');
    
    % Define the cost and its derivatives on the Grassmann manifold
    tuple.U = stiefelfactory(m, p);
    tuple.V = stiefelfactory(n, p);
    M = productmanifold(tuple);
    
    % Define a problem structure, specifying the manifold M, the cost
    % function and its derivatives.
    problem.M = M;
    
    function store = prepare(X, store)
        if ~isfield(store, 'AV')
            V = X.V;
            store.AV = A*V;
        end
        if ~isfield(store, 'AtU')
            U = X.U;
            store.AtU = A'*U;
        end
    end
    % Cost function
    problem.cost  = @cost;
    function [f, store] = cost(X, store)
        store = prepare(X, store);
        U  = X.U;
        AV = store.AV;
        f  = -(p:-1:1)*diag(U'*AV);
    end
    % Euclidean gradient of the cost function
    problem.egrad = @egrad;
    function [g, store] = egrad(X, store)
        store = prepare(X, store);
        AV  = store.AV;
        AtU = store.AtU;
        g.U = -bsxfun(@times, AV, p:-1:1);
        g.V = -bsxfun(@times, AtU, p:-1:1);
        store = incrementcounter(store, 'gradhesscalls');
    end
    % Euclidean Hessian of the cost function
    problem.ehess = @ehess;
    function [h, store] = ehess(X, H, store) %#ok<INUSL>
        Udot = H.U;
        Vdot = H.V;
        AVdot = A*Vdot;
        AtUdot = A'*Udot;
        h.U = -bsxfun(@times, AVdot, p:-1:1);
        h.V = -bsxfun(@times, AtUdot, p:-1:1);
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'gradhesscalls');
    end

    problem.name = sprintf('Truncated SVD, St(%d, %d) x St(%d, %d)', m, p, n, p);    

end
