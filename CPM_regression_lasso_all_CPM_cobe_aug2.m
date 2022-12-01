function [all_output, all_fea] = CPM_regression_lasso_all_CPM_cobe_aug2(all_mats1, behav1,...
    subName1, subName_all1, IAA1, IBB1, mats2, behav2, subName2, subName_all2, IAA2, IBB2, thresh, norm, k, alpha, comps)

    num_sub_total1 = length(subName1);
    num_sub_total2 = length(subName2);
    num_sample_total1 = length(IBB1);
    num_sample_total2 = length(IBB2);
    for i = 1: k
        variable_name{i} = ['fold', num2str(i)];
    end

    y_pred_test_all = zeros(length(comps), num_sub_total1, length(thresh), length(alpha));
    all_pred_group2 = zeros(length(comps), num_sub_total2, length(thresh), length(alpha),k);

    indices = crossvalind('Kfold',subName1,k);
    for n = 1: length(comps)
        for j = 1 : length(thresh)
            IDX_test = [];
            y_pred_test_cv = zeros(num_sample_total1, length(alpha));
            pred_group2_cv = zeros(num_sample_total2, length(alpha), k);
            for i = 1:k
                in_fold = num2str(i);
                subID_test=subName1(indices==i); subID_train=subName1(indices~=i);
                test=find(contains(subName_all1,subID_test)); 
                train=find(contains(subName_all1,subID_train));
                raw_tr_x1 = all_mats1(train,:);
                raw_tr_x2 = mats2;
                raw_tr_x = [raw_tr_x1; raw_tr_x2];
                [raw_tr_x,u1,s1] = normalize(raw_tr_x);
                raw_tr_y = behav1(train,:);
                % COBE
                n_comm=comps(n);
                gps = 10;
                idd=floor(linspace(0,size(raw_tr_x,1),gps+1));
                idd=diff(idd);
                A=mat2cell(raw_tr_x,idd,size(raw_tr_x,2));
                A=cellfun(@(c) c',A,'UniformOutput',false);
                [c,Q,~,~]=cobe_zy(A,n_comm);
                % variance explained
%                 raw_tr_x_ = c'*raw_tr_x';
%                 x_load = (raw_tr_x_* raw_tr_x) ./ diag(raw_tr_x_* raw_tr_x_');
%                 pctVar = sum(abs(x_load)*abs(x_load),1) / sum(sum(abs(raw_tr_x)*abs(raw_tr_x),1));
%                 covmat = (c'*raw_tr_x')* raw_tr_x* c; %calculate covariance matrix
%                 varE = diag(covmat) .* diag(covmat) / sum(diag(covmat) .* diag(covmat)) %calcualte covariance explained by each component
                raw_tr_x=[A{:}]-c*cell2mat(Q);
                raw_tr_x = raw_tr_x.';
                raw_tr_x = raw_tr_x(1:size(raw_tr_x1,1),:);
                fold_fea.(['test_idx', in_fold, '_fold_in']) = test;                              
                fold_fea.(['train_idx', in_fold, '_fold_in']) = train;
    
                te_x = all_mats1(test,:);
                te_x = normalize(te_x,"center",u1,"scale",s1);
                te_x = te_x'-c*c'*te_x';
                te_x = te_x';

                [r,p] = corr(raw_tr_x,raw_tr_y);
                mask = find(p<=thresh(j)==1);
                train_edges = raw_tr_x(:,mask);
                test_edges = te_x(:,mask);
                all_edges2 = normalize(mats2,"center",u1,"scale",s1);
                group2_edge = all_edges2(:,mask);
                fold_fea.(['summation_mask_rp', in_fold, '_fold_in']) = [r(mask),p(mask)]; 

                if norm
                    [tr_fea,u,s] = normalize(train_edges);
                    test_fea = normalize(test_edges,"center",u,"scale",s);
                    all_edges2_fea = normalize(group2_edge,"center",u,"scale",s);
                else
                    tr_fea = train_edges;
                    test_fea = test_edges;
                    all_edges2_fea = group2_edge;  
                end

                [B,FitInfo]=lasso(tr_fea,raw_tr_y,'lambda',alpha);
                y_pred_tr=tr_fea*B+FitInfo.Intercept;
                y_pred_test=test_fea*B+FitInfo.Intercept;
                for a = 1: length(alpha)
                    [r,p] = corr(y_pred_tr(:,a),raw_tr_y)
                end
                y_pred_test_cv(test,:) = y_pred_test; 
                IDX_test=[IDX_test;test];

                fold_output.weight = B;
                fold_output.intercept = FitInfo.Intercept;
                fold_output.cobe_w = c;
                fold_fea.(['summation_mask_', in_fold, '_fold_in']) = mask; 
    
                %%%%%%%%%%group2
                y_pred_test2=all_edges2_fea*B+FitInfo.Intercept;
                pred_group2_cv(:,:,i) = y_pred_test2;
                fold_fea.(['model_weight_', variable_name{i}]) = fold_output;
            end
%             y_pred_test_cv = y_pred_test_cv(IDX_test,:);
            for sub=1:length(subName1)
                y_pred_test_all(n,sub,j,:)=mean(y_pred_test_cv(IBB1==sub,:),1);
            end   
            for sub=1:length(subName2)
                all_pred_group2(n,sub,j,:,:)=mean(pred_group2_cv(IBB2==sub,:,:),1);
            end
            name = strsplit(num2str(thresh(j)), '.');
            name = name{2};
            one_fea.(['p_', name]) = fold_fea;
        end
        name = num2str(comps(n));
        all_fea.(['comp_feature', name]) = one_fea;
    end
    all_output.prediction_group1 = y_pred_test_all;
    all_output.prediction_group2 = all_pred_group2;
    all_output.true_group1 = behav1(IAA1);
    all_output.true_group2 = behav2(IAA2);
end

function [R2] = Rsquared(predicted_values, y)
    SStot = var(y)*length(y);
    SSres = sum( (y(:)-predicted_values(:)).^2 );
    R2 = 1 - SSres/SStot;
end
