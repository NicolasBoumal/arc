function problem = generalized_eigenvalue_computation_problem(A, B, n, p)
    
    % Generate some random data to test the function
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
    if ~exist('B', 'var') || isempty(B)
        n = size(A, 1);
        e = ones(n, 1);
        B = spdiags([-e 2*e -e], -1:1, n, n); % Symmetric positive definite
    end
    
    if ~exist('p', 'var') || isempty(p)
        p = 3;
    end

    % Make sure the input matrix is square and symmetric
    n = size(A, 1);
    assert(isreal(A), 'A must be real.')
    assert(size(A, 2) == n, 'A must be square.');
    assert(norm(A-A', 'fro') < n*eps, 'A must be symmetric.');
    assert(p <= n, 'p must be smaller than n.');
    
    % Define the cost and its derivatives on the generalized 
    % Grassmann manifold, i.e., the column space of all X such that
    % X'*B*X is identity. 
    gGr = grassmanngeneralizedfactory(n, p, B);
    
    problem.M = gGr;
    function store = prepare(X, store)
        if ~isfield(store, 'AX')
            store.AX = A*X;
        end
    end
    problem.cost = @cost;
    function [f, store] = cost(X, store)
        store = prepare(X, store);
        AX = store.AX;
        f = -.5*(X(:)'*AX(:));
    end
    problem.egrad = @egrad;
    function [G, store] = egrad(X, store)
        store = prepare(X, store);
        AX = store.AX;
        G = -AX;
    end
    problem.ehess = @ehess;
    function [H, store] = ehess(X, Xdot, store) %#ok<INUSL>
        H = -A*Xdot;
        store = incrementcounter(store, 'hesscalls');
    end

    problem.name = sprintf('Generalized eigenvalue computation, gGr(%d, %d)', n, p);

end
