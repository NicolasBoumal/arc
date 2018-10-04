function optproblem = rotation_synchronization(n, N, ERp)

% Modified from test file for synchronization of rotations using the
% maximum likelihood algorithm SynchronizeMLE described in
%
% N. Boumal, A. Singer and P.-A. Absil, 2013,
%   Robust estimation of rotations from relative measurements
%   by maximum likelihood,
% in the proceedings of the 52nd Conference on Decision and Control (CDC).
%
% Code: Nicolas Boumal, UCLouvain, 2013.
% Contact: nicolasboumal@gmail.com

% Synchronize N rotations in SO(n). Rtrue contains the true rotations we
% are looking for. Of course, these are not known for a real problem
% instance.
Rtrue = randrot(n, N);

% Rotations with indices in A are anchored. Ra contains the rotations of
% these anchors. That is: Ra(:, :, k) contains the nxn rotation matrix
% associated to the node A(k), which is anchored. If there are no anchors,
% simply let A = [1] and Ra = eye(n), that is: artificially fix any of the
% rotations to an arbitrary value.
m = 1;
A = 1:m;
Ra = Rtrue(:, :, A);

% For this test problem instance, we generate a random Erdos-Renyi graph.
% For each edge in the graph, we will have a measurement of the relative
% rotation between the adjacent nodes. The data is presented this way:
% I, J are two column vectors of length M, where M is the number of edges.
% There is an edge between nodes I(k) and J(k) for k = 1 : M.
% The graph is symmetric, so that we only define each edge once. That is:
% if the matrix [I J] contains the row [a b], then it does not contain the
% row [b a]. ERp is the edge density in the Erdos-Renyi graph.
% For a complete graph, you may use: [I J] = find(triu(ones(N), 1));
% ERp = 0.75;
[I, J] = erdosrenyi(N, ERp);
M = length(I);

% Pick noise parameters (see the mixture of Langevin distribution described
% in our CDC paper) and generate noise (random rotation matrices) according
% to these parameters. The measurements are stored in H, a 3D matrix such
% that each slice H(:, :, k) is an nxn rotation matrix which corresponds to
% a measurement of the relative rotation Ri Rj^T,
% with Ri = Rtrue(:, :, I(k)) and Rj = Rtrue(:, :, J(k)).
% The measurement H(:, :, k) is distributed around the real relative
% rotation with parameters kappa1(k), kappa2(k) and p(k), where kappa1,
% kappa2 and p are vectors of length M.
% Of course, for a real problem instance, you just need to obtain H from
% your data directly.
kappa1 = 8.0*ones(M, 1);
kappa2 = 0.0*ones(M, 1);
p = 0.85*ones(M, 1);
Z = randlangevinmixture(n, kappa1, kappa2, p);
Htrue = multiprod(Rtrue(:, :, I), multitransp(Rtrue(:, :, J)));
H = multiprod(Z, Htrue);

% Put all the data together in a structure which describes the
% synchronization problem. Here, kappa1, kappa2 and p need not be the same
% as those used in generating the measurements. In fact, for real
% applications, the true values for these noise parameters are unknown. You
% must then guess these parameters, based on prior information you have
% about the measurements (how accurate you think they are, how many
% outliers you expect there to be). If you are not sure, set them to a
% reasonable value and use MLE+: our MLE algorithm which will iteratively
% estimate the rotations then the noise parameters, then the rotations,
% etc. For MLE+, kappa1, kappa2 and p need to be constant vectors.
% Otherwise, only their average is taken into account. If you really don't
% know, try with kappa1 = 10, kappa2 = 0 and p = 0.9. This means you expect
% 10% of outliers and 90% of measurements with roughly 20 degrees errors.
synchroproblem = buildproblem_synchro(n, N, M, I, J, H, kappa1, kappa2, p, A, Ra);

optproblem = synchronizeMLE(synchroproblem);

optproblem.name = sprintf('Synchronization of %d rotations in SO(%d)', N, n);

end
