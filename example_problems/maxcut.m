function problem = maxcut(graphid, p)

    % Load Gset graph
    graph = load(sprintf('MaxCut/Gset/g%d.mat', graphid));
    A = graph.A;
    problem.name = sprintf('Max-cut with Gset graph #%d', graphid);
    
    n = size(A, 1);
    
    % Make sure A is symmetric
    assert(size(A, 2) == n);
    assert(nnz(A-A') == 0);
    
    if ~exist('p', 'var') || isempty(p)
        p = ceil(sqrt(8*n+1)/2);
    end

    manifold = obliquefactory(p, n, true);
    
    problem.M = manifold;
    

    % Products with A dominate the cost, hence we store the result.
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
        store = incrementcounter(store, 'gradhesscalls');
    end

    % If you want to, you can specify the Hessian as well.
    problem.hess = @hess;
    function [H, store] = hess(Y, Ydot, store)
        store = prepare(Y, store);
        SYdot = A*Ydot - bsxfun(@times, Ydot, store.diagAYYt);
        H = manifold.proj(Y, SYdot);
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'gradhesscalls');
    end

end
