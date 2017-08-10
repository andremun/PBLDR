function [Z,A,B,C,error,R2] = PBLDR(X, Y, opts)

errorfcn = @(alpha,X,n,m) sum(sum((X-(reshape(alpha((2*n)+1:end),m,2)*... % B,C
                                      reshape(alpha(1:2*n),2,n)...        % A
                                      *X(:,1:n)')').^2,1),2);

n = size(X, 2); % Number of features
Xbar = [X Y];
m = size(Xbar, 2);
if opts.zflag
    Xbar = zscore(Xbar);
    X = Xbar(:,1:n);
end
Hd = pdist(X)';

if opts.analytic
    Xbar = Xbar';
    X = X';
    [V,D] = eig(Xbar*Xbar');
    [~,idx] = sort(abs(diag(D)),'descend');
    V = V(:,idx(1:2));
    B = V(1:n,:);
    C = V(n+1:m,:)';
    Xr = X'/(X*X');
    A = V'*Xbar*Xr;
    Z = A*X;
    Xhat = [B*Z; C'*Z];
    error = sum(sum((Xbar-Xhat).^2,2));
    R2 = diag(corr(Xbar',Xhat')).^2;
else
    alpha = zeros(2*m+2*n, opts.ntries);
    eoptim = zeros(1, opts.ntries);
    perf = zeros(1, opts.ntries);

    cmaopts = bipopcmaes;
    cmaopts.StopFitness = 0;
    cmaopts.MaxRestartFunEvals = 0;
    cmaopts.MaxFunEvals  = opts.fevals;
    cmaopts.EvalParallel = 'no';
    initstr = ['2*rand(' num2str(2*m+2*n) ',1)-1'];

    for i=1:opts.ntries
        [alpha(:,i),eoptim(i)] = bipopcmaes(errorfcn, ...
                                           initstr, ...
                                           1, ...
                                           cmaopts, ...
                                           Xbar, ...
                                           n, ...
                                           m);
        A = reshape(alpha(1:2*n,i),2,n);
        Z = X*A';
        perf(i) = corr(Hd,pdist(Z)');
    end

    [~,idx] = max(perf);
    A = reshape(alpha(1:2*n,idx),2,n);
    Z = X*A';
    B = reshape(alpha((2*n)+1:end,idx),m,2);
    Xhat = Z*B';
    C = B(n+1:m,:)';
    B = B(1:n,:);
    error = sum(sum((Xbar-Xhat).^2,2));
    R2 = diag(corr(Xbar,Xhat)).^2;
end