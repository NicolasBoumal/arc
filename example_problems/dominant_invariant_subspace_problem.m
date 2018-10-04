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
        A = randn(n);
        A = (A+A')/2;
    end
    if ~exist('p', 'var') || isempty(p)
        p = 3;
    end
    
    % Make sure the input matrix is square and symmetric
    n = size(A, 1);
	assert(isreal(A), 'A must be real.')
    assert(size(A, 2) == n, 'A must be square.');
    assert(all(all(A == A')), 'A must be symmetric.');
	assert(p <= n, 'p must be smaller than n.');
    
    % Define the cost and its derivatives on the Grassmann manifold
    Gr = grassmannfactory(n, p);
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
        f = -.5*(X(:)'*AX(:));
    end
    problem.egrad = @egrad;
    function [G, store] = egrad(X, store)
        store = prepare(X, store);
        AX = store.AX;
        G = -AX;
        store = incrementcounter(store, 'gradhesscalls');
    end
    problem.ehess = @ehess;
    function [H, store] = ehess(X, Xdot, store) %#ok<INUSL>
        H = -A*Xdot;
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'gradhesscalls');
    end

    problem.name = sprintf('Dominant invariant subspace, Gr(%d, %d)', n, p);

end
