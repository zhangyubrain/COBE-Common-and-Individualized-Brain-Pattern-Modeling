clc; close all; clear;

rng('default'); rng(0);

n_training=200;         % number of training samples
n_test=50;              % number of test samples
d_feature=4950;         % dimensionality of features

Fe_train=randn(n_training,d_feature);
Fe_test=randn(n_test,d_feature);


%% COBE
n_comm=4;       % Number of common components
gps=10;         % Number of groups
idd=floor(linspace(0,size(Fe_train,1),gps+1));
idd=diff(idd);

% Standardize training data
[Fe_train,u1,s1] = normalize(Fe_train);

% Extract common components
A=mat2cell(Fe_train,idd,size(Fe_train,2));
A=cellfun(@(c) c',A,'UniformOutput',false);
[c,Q,~,~]=cobe_zy(A,n_comm);

% Compute covariance explained by common components
Fe_common = c'*Fe_train';                                                       % calculate common latent features
x_load = (Fe_common*Fe_train)./diag(Fe_common*Fe_common');
% pctVar = sum(abs(x_load)*abs(x_load),1) / sum(sum(abs(raw_tr_x)*abs(raw_tr_x),1));
covmat = (c'*Fe_train')* Fe_train* c;                                           % calculate covariance matrix
varE = diag(covmat) .* diag(covmat) / sum(diag(covmat) .* diag(covmat));        % calcualte covariance explained by each component

% Extract individualized features on the training set by removing common features
Fe_train=[A{:}]-c*c'*Fe_train';                                                 % or equivalently Fe_train=[A{:}]-c*cell2mat(Q);
Fe_train=Fe_train';

% Extract individualized features on the test set
Fe_test=normalize(Fe_test,"center",u1,"scale",s1);
Fe_test=Fe_test'-c*c'*Fe_test';
Fe_test=Fe_test';


%% Compare common components extracted by COBE and components by PCA
[coeff,score]=pca([A{:}]','NumComponents',n_comm);
figure;
subplot(1,3,1); imagesc(c); title('COBE'); xlabel('Components'); ylabel('Features');
set(gca,'xtick',1:n_comm,'xticklabel',1:n_comm,'fontsize',15);
subplot(1,3,2); imagesc(coeff); title('PCA'); xlabel('Components'); ylabel('Features');
set(gca,'xtick',1:n_comm,'xticklabel',1:n_comm,'fontsize',15);
a=corr(c,coeff); subplot(1,3,3); imagesc(a); colorbar; clim([-1 1]);
xlabel('PCA components'); ylabel('COBE components'); title('Correlation');
set(gca,'xtick',1:n_comm,'xticklabel',1:n_comm,'ytick',1:n_comm,'yticklabel',1:n_comm,'fontsize',15);
