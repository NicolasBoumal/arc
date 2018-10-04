function problem = low_rank_matrix_completion_problem(m, n, k)
    
    % Random data generation. First, choose the size of the problem.
    % We will complete a matrix of size mxn of rank k:
    if ~exist('m', 'var') || isempty(m)
        m = 200;
    end
    if ~exist('n', 'var') || isempty(n)
        n = 500;
    end
    if ~exist('k', 'var') || isempty(k)
        k = min([m, n, 10]);
    end

    % Generate a random mxn matrix A of rank k
    L = randn(m, k);
    R = randn(n, k);
    A = L*R';
    % Generate a random mask for observed entries: P(i, j) = 1 if the entry
    % (i, j) of A is observed, and 0 otherwise.
    fraction = 4 * k*(m+n-k)/(m*n);
    P = sparse(rand(m, n) <= fraction);
    % Hence, we know the nonzero entries in PA:
    PA = P.*A;

    % Pick the manifold of matrices of size mxn of fixed rank k.
    problem.M = fixedrankembeddedfactory(m, n, k);

    % Define the problem cost function. The input X is a structure with
    % fields U, S, V representing a rank k matrix as U*S*V'.
    % f(X) = 1/2 * || P.*(X-A) ||^2
    problem.cost = @cost;
    function f = cost(X)
        % Note that it is very much inefficient to explicitly construct the
        % matrix X in this way. Seen as we only need to know the entries
        % of Xmat corresponding to the mask P, it would be far more
        % efficient to compute those only.
        Xmat = X.U*X.S*X.V';
        f = .5*norm( P.*Xmat - PA , 'fro')^2;
    end

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a standard function of X.
    % nabla f(X) = P.*(X-A)
    problem.egrad = @egrad;
    function G = egrad(X)
        % Same comment here about Xmat.
        Xmat = X.U*X.S*X.V';
        G = P.*Xmat - PA;
    end

    % This is optional, but it's nice if you have it.
    % Define the Euclidean Hessian of the cost at X, along H, where H is
    % represented as a tangent vector: a structure with fields Up, Vp, M.
    % This is the directional derivative of nabla f(X) at X along Xdot:
    % nabla^2 f(X)[Xdot] = P.*Xdot
    problem.ehess = @euclidean_hessian;
    function [ehess, store] = euclidean_hessian(X, H, store)
        % The function tangent2ambient transforms H (a tangent vector) into
        % its equivalent ambient vector representation. The output is a
        % structure with fields U, S, V such that U*S*V' is an mxn matrix
        % corresponding to the tangent vector H. Note that there are no
        % additional guarantees about U, S and V. In particular, U and V
        % are not orthonormal.
        ambient_H = problem.M.tangent2ambient(X, H);
        Xdot = ambient_H.U*ambient_H.S*ambient_H.V';
        % Same comment here about explicitly constructing the ambient
        % vector as an mxn matrix Xdot: we only need its entries
        % corresponding to the mask P, and this could be computed
        % efficiently.
        ehess = P.*Xdot;
        store = incrementcounter(store, 'hesscalls');
    end

    problem.name = sprintf('Low rank matrix completion, FixedRankEmbedded(%d, %d, %d)', m, n, k);    

end