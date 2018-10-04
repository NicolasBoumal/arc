function optproblem = lrmc_grassmann(m, n, r, osf)
% Completion problem for a random matrix of size mxn and rank r, with
% number of entries observed being osf (oversampling factor) times the
% number of degrees of freedom of a matrix of that size and rank.

% Modified from code for the RTRMC/RCGMC algorithm (low-rank matrix completion).
% Algorithm by Nicolas Boumal and P.-A. Absil.
% Code by Nicolas Boumal, UCLouvain, April 2, 2013.
%
% http://perso.uclouvain.be/nicolas.boumal/RTRMC/
% http://sites.uclouvain.be/absil/RTRMC/
%
% See also: rtrmc buildproblem initialguess

% If this script fails, try executing installrtrmc.m, to compile the mex files.

    %% Problem instance generation

    % Dimensions of the test problem
    k = osf*r*(m+n-r);                       % number of known entries

    % Generate an m-by-n matrix of rank true_rank in factored form: A*B
    true_rank = r;
    A = randn(m, true_rank)/true_rank.^.25;
    B = randn(true_rank, n)/true_rank.^.25;

    % Pick k (or about k) entries uniformly at random
    [I, J, k] = randmask(m, n, k);

    % Compute the values of AB at these entries
    % (this is a C-Mex function)
    X = spmaskmult(A, B, I, J);

    % Define the confidence we have in each measurement X(i)
    C = ones(size(X));

    % Add noise if desired
    noisestd = 0;
    X = X + noisestd*randn(size(X));


    %% Format problem into Manopt problem structure

    % Pick a value for lambda, the regularization parameter
    lambda = 0;


    perm = randperm(k);
    I = I(perm);
    J = J(perm);
    X = X(perm);
    C = C(perm);


    % Build a problem structure
    lrmcproblem = buildproblem_lrmc(I, J, X, C, m, n, r, lambda);

    % Call the algorithm here. The outputs are U and W such that U is
    % orthonormal and the product U*W is the matrix estimation. The third
    % output, stats, is a structure array containing lots of information about
    % the iterations made by the optimization algorithm. In particular, timing
    % information should be retrieved from stats, as in the code below.
    optproblem = rtrmc(lrmcproblem, false);

    optproblem.name = sprintf('LRMC on Grassmann, %dx%d, rank %d, osf %d', m, n, r, osf);
    
end
