function problem = elliptope_SDP_complex_problem(A, n, p)

    % If no inputs are provided, since this is an example file, generate
    % a random complex matrix. This is for illustration purposes only.
    if ~exist('n', 'var') || isempty(n)
        if ~exist('A', 'var') || isempty(A)
            n = 128;
        else
            n = size(A, 1);
        end
    end
    if ~exist('A', 'var') || isempty(A)
        A = randn(n) + 1i*randn(n);
        A = (A+A')/sqrt(2*n);
    end

    n = size(A, 1);
    assert(n >= 2, 'A must be at least 2x2.');
    assert(size(A, 2) == n, 'A must be square.');
    
    % Force A to be Hermitian
    A = (A+A')/2;
    
    % By default, pick a sufficiently large p (number of columns of Y).
    if ~exist('p', 'var') || isempty(p)
        p = floor(sqrt(n)+1);
    end
    
    assert(p >= 1 && p == round(p), 'p must be an integer >= 1.');

    % Pick the manifold of complex n-by-p matrices with unit norm rows.
    manifold = obliquecomplexfactory(p, n, true);
    
    problem.M = manifold;

    % Products with A dominate the cost, hence we store the result.
    % This allows to share the results among cost, grad and hess.
    % This is completely optional.
    function store = prepare(Y, store)
        if ~isfield(store, 'AY')
            AY = A*Y;
            store.AY = AY;
            store.diagAYYt = sum(real(AY .* conj(Y)), 2);
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

    problem.name = sprintf('Elliptope SDP Complex, Complex OB(%d, %d)', n, p);

end
