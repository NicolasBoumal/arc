function problem = shapefit_leastsquares(n, d)
% We estimate n points in R^d; returns a least-squares version of ShapeFit.

    % Generic useful functions
    center_cols = @(A) bsxfun(@minus, A, mean(A, 2));
    normalize_cols = @(A) bsxfun(@times, A, 1./sqrt(sum(A.^2, 1)));

    % Those points are the columns of T : they are what we need to estimate,
    % up to scaling and translation. We center T for convenience.
    T = center_cols(rand(d, n));

    % We get a measurement of some pairs of relative directions.
    % Which pairs is encoded in this graph, with J being the (signed,
    % transposed) incidence matrix. J is n x m, sparse.
    edge_fraction = 0.1;
    [ii, jj] = erdosrenyi(n, edge_fraction);
    m = length(ii);
    J = sparse([ii ; jj], [(1:m)' ; (1:m)'], [ones(m, 1), -ones(m, 1)], n, m, 2*m);

    % The measurements give us the directions from one point to another.
    % That is: we get the position difference, normalized.
    % Here, with Gaussian noise. Least-squares will be well-suited for this.
    sigma = .1;
    V = normalize_cols(T*J + sigma*randn(d, m)); % d x m
    
    % Outliers
    outlier_fraction = .3;
    outliers = rand(1, m) < outlier_fraction;
    V(:, outliers) = normalize_cols(randn(d, sum(outliers)));

    VJt = full(V*J');

    problem.M = shapefitfactory(VJt);

    % This linear operator computes the orthogonal projection of each
    % difference ti - tj on the orthogonal space to v_ij.
    % If the alignment is compatible with the data, then this is zero.
    % A(T) is a d x m matrix.
    A = @(T) T*J - bsxfun(@times, V, sum(V .* (T*J), 1));

    % Need the adjoint of A, too. Input is dxm, output is dxn.
    Astar = @(W) (W - bsxfun(@times, V, sum(V.*W, 1)))*J';

    % This is a least-squares formulation of the problem.
    % That is, we minimize a (very nice) convex cost over an affine space.
    % Since the smooth solvers in Manopt converge to critical points, this
    % means they converge to global optimizers.
    problem.cost  = @(T) 0.5*norm(A(T), 'fro')^2;
    problem.egrad = @egrad;
    function [G, store] = egrad(T, store)
        G = Astar(A(T)); % this call to A(T) could be stored from that of cost
        store = incrementcounter(store, 'gradhesscalls');
    end
    problem.ehess = @ehess;
    function [H, store] = ehess(T, Tdot, store) %#ok<INUSL>
        H = Astar(A(Tdot));
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'gradhesscalls');
    end

    problem.name = sprintf('ShapeFit least-squares for %d points in R^{%d}', n, d);
    
end
