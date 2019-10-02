close all
clear

title_names = {'q_0','q_1','q_2','q_3','x_L','y_L','x_R','y_R','p','s','u_0','u_1','u_2','u_3'};


Dir=dir('./run_*');
results=nan(2,4,length(Dir));
results_gdoc=nan(2,3,length(Dir));

%%
disp('--------  ---------')

for rep=1:length(Dir)

    load([Dir(rep).name '/results/mvae_final_completeloss_test1.mat'])
ref=x_sample;
err_droniou_martina_test1 = immse(double(x_reconstruct),ref);
err_droniou_martina_test1_perc = err_droniou_martina_test1/4;
results(1,1,rep)=err_droniou_martina_test1;
results(2,1,rep)=err_droniou_martina_test1_perc;

load([Dir(rep).name '/results/mvae_final_completeloss_test2.mat'])
err_droniou_martina_test2 = immse(double(x_reconstruct),ref);
err_droniou_martina_test2_perc = err_droniou_martina_test2/4;
results(1,4,rep)=err_droniou_martina_test2;
results(2,4,rep)=err_droniou_martina_test2_perc;

results_gdoc(1,1,rep)= immse(double(x_reconstruct(:,end-3:end)),ref(:,end-3:end));
results_gdoc(2,1,rep)= results_gdoc(1,1,rep)/4;

load([Dir(rep).name '/results/mvae_final_completeloss_test3.mat'])
err_droniou_martina_test3 = immse(double(x_reconstruct),ref);
err_droniou_martina_test3_perc = err_droniou_martina_test3/4;
results(1,3,rep)=err_droniou_martina_test3;
results(2,3,rep)=err_droniou_martina_test3_perc;
index=[1 2 3 4 9 10 11 12 17 19];
results_gdoc(1,3,rep)= immse(double(x_reconstruct(:,index)),ref(:,index));
results_gdoc(2,3,rep)= results_gdoc(1,3,rep)/4;

load([Dir(rep).name '/results/mvae_final_completeloss_test4.mat'])
err_droniou_martina_test4 = immse(double(x_reconstruct),ref);
err_droniou_martina_test4_perc = err_droniou_martina_test4/4;
results(1,2,rep)=err_droniou_martina_test4;
results(2,2,rep)=err_droniou_martina_test4_perc;

results_gdoc(1,2,rep)= immse(double(x_reconstruct(:,index)),ref(:,index));
results_gdoc(2,2,rep)= results_gdoc(1,2,rep)/4;

end
disp('-------- case_based ---------')
results
disp('-------- googledoc_based ---------')
results_gdoc

Qresults = quantile(results,[.05 .25 .50 .75 .95],3)
Qresults_gdoc = quantile(results_gdoc,[.05 .25 .50 .75 .95],3)