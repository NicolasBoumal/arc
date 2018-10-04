function problem = elliptope_SDP_problem(A, n, p)

    % If no inputs are provided, since this is an example file, generate
    % a random Erdos-Renyi graph. This is for illustration purposes only.
    if ~exist('n', 'var') || isempty(n)
        if ~exist('A', 'var') || isempty(A)
            n = 128;
        else
            n = size(A, 1);
        end
    end
    if ~exist('A', 'var') || isempty(A)
        A = triu(rand(n) <= .1, 1);
        A = (A+A.')/(2*n);
    end

    n = size(A, 1);
    assert(n >= 2, 'A must be at least 2x2.');
    assert(isreal(A), 'A must be real.');
    assert(size(A, 2) == n, 'A must be square.');
    
    % Force A to be symmetric
    A = (A+A.')/2;
    
    % By default, pick a sufficiently large p (number of columns of Y).
    if ~exist('p', 'var') || isempty(p)
        p = ceil(sqrt(8*n+1)/2);
    end
    
    assert(p >= 2 && p == round(p), 'p must be an integer >= 2.');

    % Pick the manifold of n-by-p matrices with unit norm rows.
    manifold = obliquefactory(p, n, true);
    
    problem.M = manifold;

    % Products with A dominate the cost, hence we store the result.
    % This allows to share the results among cost, grad and hess.
    % This is completely optional.
    function store = prepare(Y, store)
        if ~isfield(store, 'AY')
            AY = A*Y;
            store.AY = AY;
            store.diagAYYt = sum(AY .* Y, 2);
        end
    end
    
    % Define the cost function to be /minimized/.
    problem.cost = @cost;
    function [f, store] = cost(Y, store)
        store = prepare(Y, store);
        f = .5*sum(store.diagAYYt);
    end

    % Define the Riemannian gradient.
    problem.grad = @grad;
    function [G, store] = grad(Y, store)
        store = prepare(Y, store);
        G = store.AY - bsxfun(@times, Y, store.diagAYYt);
    end

    % If you want to, you can specify the Riemannian Hessian as well.
    problem.hess = @hess;
    function [H, store] = hess(Y, Ydot, store)
        store = prepare(Y, store);
        SYdot = A*Ydot - bsxfun(@times, Ydot, store.diagAYYt);
        H = manifold.proj(Y, SYdot);
        store = incrementcounter(store, 'hesscalls');
    end

    problem.name = sprintf('Elliptope SDP, OB(%d, %d)', n, p);

end
