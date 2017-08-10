getfromfile = @(filename,varname) getfield(load(filename,varname),varname);
ucidata = getfromfile('ucidata_final.mat','ucidata');
selvars = [14 15 19 22 28 33 34 36 38 41];
algoperf = 44:53;
Xbar = ucidata(:,[selvars algoperf]);
n = length(selvars);
m = size(Xbar,2);
MASK_FEAT = 1:n;
MASK_PERF = n+1:m;
MASK_SQRT = [2 3 4 5 6 8 9 MASK_PERF];
MASK_ATANH = 7;

Xbar(:,MASK_ATANH) = atanh(0.99999.*(2.*Xbar(:,MASK_ATANH)-1));
Xbar(:,MASK_SQRT) = sqrt(Xbar(:,MASK_SQRT));

Xbar = zscore(Xbar);
X = Xbar(:,MASK_FEAT);
Y = Xbar(:,MASK_PERF);

opts.zflag = false;
opts.ntries = 1;
opts.fevals = 1e4;
opts.analytic = true;
[a.Z,a.A,a.B,a.C,a.error,a.R2] = core(X,Y,opts);

opts.analytic = false;
[b.Z,b.A,b.B,b.C,b.error,b.R2] = PBLDR(X,Y,opts);
