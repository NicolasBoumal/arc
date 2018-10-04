function problem = dominant_invariant_subspace_problem(A, n, p)
    
    % Generate some random data to test the function
    if ~exist('n', 'var') || isempty(n)
        if ~exist('A', 'var') || isempty(A)
            n = 128;
        else
            n = size(A, 1);
        end
    end
    if ~exist('A', 'var') || isempty(A)
        A = randn(n) + 1i*randn(n);
        A = (A+A')/2;
    end
    if ~exist('p', 'var') || isempty(p)
        p = 3;
    end
    
    % Make sure the input matrix is Hermitian
    n = size(A, 1);
    assert(size(A, 2) == n, 'A must be square.');
    assert(norm(A-A', 'fro') < n*eps, 'A must be Hermitian.');
    assert(p<=n, 'p must be smaller than n.');
    
    % Define the cost and its derivatives on the complex Grassmann manifold
    Gr = grassmanncomplexfactory(n, p);
    problem.M = Gr;
    function store = prepare(X, store)
        if ~isfield(store, 'AX')
            store.AX = A*X;
        end
    end
    problem.cost = @cost;
    function [f, store] = cost(X, store)
        store = prepare(X, store);
        AX = store.AX;
        f = -real(trace(X'*AX));
    end
    problem.egrad = @egrad;
    function [G, store] = egrad(X, store)
        store = prepare(X, store);
        AX = store.AX;
        G = -2*AX;
    end
    problem.ehess = @ehess;
    function [H, store] = ehess(X, Xdot, store) %#ok<INUSL>
        H = -2*A*Xdot;
        store = incrementcounter(store, 'hesscalls');
    end

    problem.name = sprintf('Dominant invariant subspace complex, Complex Gr(%d, %d)', n, p);

end
