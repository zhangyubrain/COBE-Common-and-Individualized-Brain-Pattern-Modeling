close all
seeds = 1:10;
outlier = num2str(0.1)

for s = 1:length(seeds)
    seed = seeds(s);
    rng(seed);
    
    seed = num2str(seed);
    week8_data_set = {'week8_interpolation'};
    interplo_logic_set = [1];
    fconnect = {'pear'};

    n_comp = [1,2,3,4,5,6,7,8];
%     n_comp = [8];
    % outlier = {'0.05'};
    for c_ = 1: length(fconnect)
        if fconnect{c_} == 'part'
            conn = 10;
        else
            conn = 9;
        end
        for a = 1: length(interplo_logic_set)
            week8_data_type = week8_data_set{a};
  
            ser_file = ['F:\PHD\learning\project\CPM_multi_run_concate\data\week8_interpolation\100roi\2_run_no_regress_all\ser_all_fmri_rs_allconfound_full_continues.mat'];
            pla_file = ['F:\PHD\learning\project\CPM_multi_run_concate\data\week8_interpolation\100roi\2_run_no_regress_all\pla_all_fmri_rs_allconfound_full_continues.mat']; 
            load(ser_file);
            load(pla_file);  

            ser_fmri_ts2 = ser_fmri.s_im_fmri2(:,:,:,1);
            ser_fmri_ts1 = ser_fmri.s_im_fmri1(:,:,:,1);
            ser_id1 = ser_fmri.s_pcd_id1;
            ser_id2 = ser_fmri.s_pcd_id2;
            
            pla_fmri_ts2 = pla_fmri.p_im_fmri2(:,:,:,1);
            pla_fmri_ts1 = pla_fmri.p_im_fmri1(:,:,:,1);
            pla_id1 = pla_fmri.p_pcd_id1;
            pla_id2 = pla_fmri.p_pcd_id2;
            
            ser_pear_fc1 = zeros(length(ser_id1),4950);
            for idx =1:length(ser_id1)
                fc = squareform(tril(ser_fmri_ts1(:,:,idx), -1));
                ser_pear_fc1(idx,:) = fc;
            end    
            ser_pear_fc2 = zeros(length(ser_id2),4950);
            for idx =1:length(ser_id2)
                fc = squareform(tril(ser_fmri_ts2(:,:,idx), -1));
                ser_pear_fc2(idx,:) = fc;
            end   
            pla_pear_fc1 = zeros(length(pla_id1),4950);
            for idx =1:length(pla_id1)
                fc = squareform(tril(pla_fmri_ts1(:,:,idx), -1));
                pla_pear_fc1(idx,:) = fc;
            end    
            pla_pear_fc2 = zeros(length(pla_id2),4950);
            for idx =1:length(pla_id2)
                fc = squareform(tril(pla_fmri_ts2(:,:,idx), -1));
                pla_pear_fc2(idx,:) = fc;
            end 
            ser_pear_fmri=[ser_pear_fc1;ser_pear_fc2]; 
            pla_pear_fmri=[pla_pear_fc1;pla_pear_fc2];
            pla_pear_fmri = atanh(pla_pear_fmri.').';
            pla_pear_fmri=zscore(pla_pear_fmri')';
            ser_pear_fmri = atanh(ser_pear_fmri.').';
            ser_pear_fmri=zscore(ser_pear_fmri')';

            ser_hamd_week0 = ser_fmri.s_pcd_w0_hdrs;
            pla_hamd_week0 = pla_fmri.p_pcd_w0_hdrs;
            ser_hamd_week8 = ser_fmri.s_pcd_w8_hdrs;
            pla_hamd_week8 = pla_fmri.p_pcd_w8_hdrs;
            ser_hamd_diff = (ser_hamd_week0 - ser_hamd_week8)';
            pla_hamd_diff = (pla_hamd_week0 - pla_hamd_week8)';
              
            pla_subName1=cellstr(pla_id1); pla_subName2=cellstr(pla_id2); pla_subName_all=[pla_subName1;pla_subName2];
            [IC,IA,IB]=intersect(pla_subName2,pla_subName1,'stable');
            pla_hamd_diff=[pla_hamd_diff;pla_hamd_diff(IA)];        
            ser_subName1=cellstr(ser_id1); ser_subName2=cellstr(ser_id2); ser_subName_all=[ser_subName1;ser_subName2];
            [IC,IA,IB]=intersect(ser_subName2,ser_subName1,'stable');
            ser_hamd_diff=[ser_hamd_diff;ser_hamd_diff(IA)];
%             idrm = Remove_bad_samples(round(str2double(outlier)*size(ser_pear_fmri,1)), ser_pear_fmri, ser_hamd_diff);
%             ser_pear_fmri(idrm,:)=[]; ser_hamd_diff(idrm)=[]; ser_subName_all(idrm)=[];  
%             
%             idrm = Remove_bad_samples(round(str2double(outlier)*size(pla_pear_fmri,1)), pla_pear_fmri, pla_hamd_diff);
%             pla_pear_fmri(idrm,:)=[]; pla_hamd_diff(idrm)=[]; pla_subName_all(idrm)=[];

            [pla_subName,pla_IAA,pla_IBB]=unique(pla_subName_all,'stable');
            [ser_subName,ser_IAA,ser_IBB]=unique(ser_subName_all,'stable');

%             p_value = [0.1, 0.05, 0.01, 0.005, 0.001]; 
            p_value = [0.05]; 
            alpha = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];
            
            site_effect_set = [0];
            model = 'Lasso';
            normalization =[0];
            for b = 1 : length(normalization)
                if normalization(b) == 1
                    norm = 'norm_true';
                elseif normalization(b) == 0
                    norm = 'norm_false';
                end
                for i = 1:length(site_effect_set)
                    if site_effect_set(i)
                        regressout_siteffect([ser_fmri.s_pcd_site ser_site], ser_pear_fmri);
                        regressout_siteffect(pla_fmri.p_pcd_site, pla_pear_fmri1);        
                        regressout_siteffect(pla_fmri.p_pcd_site(pla_id1_id2_map(:,2)~=0), pla_pear_fmri2);        
                        plot_path = ['/home/kaz220/CPM/matlab/aug_yu/multi/norm_fish/cobe_10/result/', seed,  '/',...
                            fconnect{c_}, '/RoiToRoi/',...
                            week8_data_type, '/site_effect_remove/', norm, '/', outlier, '/', model];
                    else
                        plot_path = ['/home/kaz220/CPM/matlab/aug_yu/multi/norm_fish/cobe_10/result/', seed,  '/',...
                            fconnect{c_}, '/RoiToRoi/',...
                            week8_data_type, '/site_effect_keep/', norm, '/', outlier, '/', model];
                    end
                    if isfolder(plot_path)==0
                         mkdir(plot_path) 
                    end

                    result_file = [plot_path, '/Pla_predicted_10fold_CPM.mat'];
                    if isfile(result_file)
                        fprintf('result data: %s.\n',result_file);
                        continue
                    else
                        fprintf('will save result data: %s.\n',result_file);   
                    end
                    [all_output, all_fea] = CPM_regression_lasso_all_CPM_cobe_aug2(pla_pear_fmri, pla_hamd_diff, pla_subName,...
                        pla_subName_all, pla_IAA, pla_IBB, ser_pear_fmri, ser_hamd_diff,...
                        ser_subName, ser_subName_all, ser_IAA, ser_IBB, p_value, normalization(b), 10, alpha, n_comp); 
                    for n = 1:10                        
                        cobe_w = all_fea.comp_feature8.p_05.(['model_weight_fold', num2str(n)]).cobe_w;
                        all_w(:,:,n,s) = cobe_w;
                    end
%                     for n = 1: length(n_comp)
%                         for j = 1: length(p_value)
%                             for k = 1:length(alpha)
%                                 name1 = ['Pla -> Pla binary regression p = ', num2str(p_value(j))];
%                                 name2 = ['Pla -> Ser binary regression p = ', num2str(p_value(j))];
%                                 plot_file1 = [plot_path, '/pla_pla_predicted_10fold_cobe_comp', num2str(n_comp(n)),...
%                                     'p==', num2str(p_value(j)), 'CPM_', num2str(alpha(k)), '.jpg'];
%                                 plot_file2 = [plot_path, '/pla_ser_predicted_10fold_cobe_comp', num2str(n_comp(n)),...
%                                     'p==', num2str(p_value(j)), 'CPM_', num2str(alpha(k)), '.jpg'];
%                                 pred = all_output.prediction_group1;
%                                 pred1 = pred(n,:,j,k);
%                                 idx = find(pred1==0);
%                                 pred1(idx) = [];
%                                 true1 = all_output.true_group1;
%                                 true1(idx) = [];
%                                 rsq = Rsquared(pred1, true1)
% %                                 if rsq>0.15
% %                                     pred2_ = all_output.prediction_group2;
% %                                     pred2 = mean(pred2_, 5);
% %                                     pred2 = pred2(n,:,j,k); 
% % %                                     true1 = all_output.true_group1;
% %                                     true2 = all_output.true_group2;
% %                                     scatter_plot_box(pred1.', true1, name1, plot_file1);
% %                                     scatter_plot_box(pred2.', true2, name2, plot_file2);
% %                                 end
%                             end
%                         end
%                     end
%                     save(result_file, 'all_output', 'all_fea'); 
%                     close all
                end
            end
        end
    end
    s
end
