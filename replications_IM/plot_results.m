close all
clear

title_names = {'q_0','q_1','q_2','q_3','x_L','y_L','x_R','y_R','p','s','u_0','u_1','u_2','u_3'};


Dir=dir('./run_*');
%results=nan(2,4,length(Dir));
results_gdoc=nan(2,3,length(Dir));

%%
disp('--------  ---------')

for rep=1:length(Dir)

    
disp('-------- FM_compl_data ---------')

load([Dir(rep).name '/results/ff_fm_compl_data_test.mat'])
y_test_fm = y_test;
y_ff_fm = y_ff_fm;
err_fm = immse(y_test,y_ff_fm); fprintf('%6.4f\n', err_fm);
err_fm_perc = err_fm/4; fprintf('%6.4f\n', err_fm_perc);

results_gdoc(1,2,rep)=err_fm;
results_gdoc(2,2,rep)=err_fm_perc;


disp('-------- IM_compl_data ---------')

load([Dir(rep).name '/results/ff_im_compl_data_test.mat'])
y_test_im = y_test;
y_ff_im = y_ff_im;
err_im = immse(y_test_im,y_ff_im); fprintf('%6.4f\n', err_im);
err_im_perc = err_im/4; fprintf('%6.4f\n', err_im_perc);

results_gdoc(1,1,rep)=err_im;
results_gdoc(2,1,rep)=err_im_perc;


disp('-------- IM FM concat_compl_data ---------')

load([Dir(rep).name '/results/ff_imfm_compl_data_test.mat'])
y_test_imfm = y_test;
y_ff_imfm = y_ff_imfm;
immse(y_test_imfm(:,end-1),y_ff_imfm(:,end-1))
err_imfm = immse(y_test_imfm,y_ff_imfm); fprintf('err imfm conct %6.4f\n', err_imfm);
err_imfm_perc = 100*err_imfm/4; fprintf('err imfm conct  perc %6.4f \n', err_imfm_perc);


results_gdoc(1,3,rep)=err_imfm;
results_gdoc(2,3,rep)=err_imfm_perc;



disp('-------- IM CL task_compl_data ---------')

load([Dir(rep).name '/results/ff_im_cl_compl_data_test.mat'])
y_test_im = y_test;
y_ff_im = y_ff_im;
err_im = immse(y_test_im,y_ff_im); fprintf('%6.4f\n', err_im);
err_im_perc = err_im/4; fprintf('%6.4f\n', err_im_perc);



end
disp('-------- case_based ---------')
%results
disp('-------- googledoc_based ---------')
results_gdoc

%Qresults = quantile(results,[.05 .25 .50 .75 .95],3)
Qresults_gdoc = quantile(results_gdoc,[.05 .25 .50 .75 .95],3)