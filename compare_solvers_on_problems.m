% Generic helper to compare various solvers (possibly with various options)
% on various problems, in Manopt.
%
% First version: August 10, 2018
%
% Naman Agarwal, Nicolas Boumal, Brian Bullins, Coralia Cartis
% https://github.com/NicolasBoumal/arc

clear; clc;

% Fix randomness once and for all
rng(2018);

cd example_problems;
addpath(genpath(pwd()));

%% Build a collection of problems
problems = { ...
             dominant_invariant_subspace_problem([],  512, 12), ...
             truncated_svd_problem([], 168, 240, 20), ...
             lrmc_grassmann(2000, 5000, 10, 4), ...
             maxcut(22), ...
             rotation_synchronization(3, 15, .75), ...
             shapefit_leastsquares(500, 3), ...
};
         
nproblems = numel(problems);

% Each problem structure must have a 'name' field (a string) for display.

% Pick one initial guess for each problem (will be passed to all solvers).
% One possible improvement to this code would be to allow for more than one
% initial guess per problem, to show agregated statistics.
inits = cell(size(problems));
for P = 1 : nproblems
    inits{P} = problems{P}.M.rand();
end

% Build a collection of solvers, together with accompanying options
% structures. You can have the same solver multiple times with different
% options if that's relevant. Notice that we add the 'name' field, which is
% used to display results.
solvers_and_options = {struct('solver', @trustregions, 'name', 'RTR'), ...
                       struct('solver', @arc, 'theta', .5, 'name', 'ARC \theta = .5'), ...
                 ... % struct('solver', @arc, 'theta', 50, 'name', 'ARC \theta = 50'), ...
                 ... % struct('solver', @rlbfgs, 'name', 'RLBFGS'), ...
                 ... % struct('solver', @conjugategradient, 'beta_type', 'F-R', 'name', 'CG-FR'), ...
                 ... % struct('solver', @conjugategradient, 'beta_type', 'P-R', 'name', 'CG-PR'), ...
                       struct('solver', @conjugategradient, 'beta_type', 'H-S', 'maxiter', 10000, 'name', 'CG-HS'), ...
                 ... % struct('solver', @conjugategradient, 'beta_type', 'H-Z', 'name', 'CG-HZ'), ...
                 ... % struct('solver', @barzilaiborwein, 'name', 'BB'), ...
                 ... % struct('solver', @steepestdescent, 'name', 'GD'), ...
                       };
nsolvers = numel(solvers_and_options);
                   
% Add common options to all
for S = 1 : nsolvers
    solvers_and_options{S}.statsfun = statsfunhelper(statscounters({'hesscalls', 'gradhesscalls'}));
    solvers_and_options{S}.tolgradnorm = 1e-8;
    solvers_and_options{S}.verbosity = 0;
end

%% Run all solvers on all problems

% Reminder: when benchmarking computation time, it is important to:
%  1) Use a dedicated computer (or at least minimize other running programs)
%  2) Run the code once without recording (so that Matlab will JIT the
%     code, that is, use just-in-time compilation), then run a second time
%     to actually collect data.

infos = cell(nproblems, nsolvers);
for P = 1 : nproblems
    fprintf('Solving %s\n', problems{P}.name);
    for S = 1 : nsolvers
        fprintf('\twith %s', solvers_and_options{S}.name);
        [x, cost, info] = manoptsolve(problems{P}, inits{P}, solvers_and_options{S});
        infos{P, S} = info;
        fprintf('.\n');
    end
end

cd ..;

idstring = datestr(now(), 'mmm_dd_yyyy_HHMMSS');

%% Plot results
subplot_rows = 3;
subplot_cols = 2;
assert(subplot_rows * subplot_cols >= nproblems, ...
       sprintf('Choose subplot size to fit all %d problems.', nproblems));
xmetric = {'iter',     'time',     'gradhesscalls'};
xscale  = {'linear',   'linear',   'linear'};
ymetric = {'gradnorm', 'gradnorm', 'gradnorm'};
yscale  = {'log',      'log',      'log'};
axisnames.iter = 'Iteration #';
axisnames.time = 'Time [s]';
axisnames.gradhesscalls = '# gradient calls and Hessian-vector products';
axisnames.gradnorm = 'Gradient norm';
nmetrics = numel(xmetric);
assert(numel(ymetric) == nmetrics);
for metric = 1 : nmetrics
    figure(metric);
    clf;
    set(gcf, 'Color', 'w');
    for P = 1 : nproblems
        subplot(subplot_rows, subplot_cols, P);
        title(problems{P}.name);
        hold all;
        for S = 1 : nsolvers
            plot([infos{P, S}.(xmetric{metric})], ...
                 [infos{P, S}.(ymetric{metric})], ...
                 'DisplayName', solvers_and_options{S}.name, ...
                 'Marker', '.', 'MarkerSize', 15);
        end
        hold off;
        set(gca, 'XScale', xscale{metric});
        set(gca, 'YScale', yscale{metric});
        if P == 1
		    legend('show');
        end
        if ismember(P, [1, 3, 5])
            ylabel(axisnames.(ymetric{metric}));
        end
        if ismember(P, [5, 6])
            xlabel(axisnames.(xmetric{metric}));
        end
        grid on;
        
        % HAND TUNING
        if metric <= 3 && P == 1, ylim([1e-12, 1e2]); end
        if metric <= 3 && P == 2, ylim([1e-12, 1e4]); end
        if metric <= 3 && P == 3, ylim([1e-12, 1e0]); end
        if metric <= 3 && P == 4, ylim([1e-12, 1e4]); end
        if metric <= 3 && P == 5, ylim([1e-12, 1e0]); end
        if metric <= 3 && P == 6, ylim([1e-15, 1e2]); end
        if metric <= 3
            set(gca, 'YTick', [1e-15, 1e-10, 1e-5, 1e0]);
        end
        
    end
    figname = sprintf('compare_solvers_%s_%s_%s', idstring, xmetric{metric}, ymetric{metric});
	savefig([figname, '.fig']);
    pdf_print_code(gcf, [figname, '.pdf'], 14);
end
