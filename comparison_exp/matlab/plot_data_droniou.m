close all
clear


name = 'mvae_final' %'mvae' %'mvae4j4vnosampling' %

color = lines(2);

load(['../results/',name,'_test1.mat'])
x_sample_1 = x_sample;
x_reconstruct_1 = x_reconstruct;
%x_reconstruct_var_1 = sqrt(exp(x_reconstruct_log_sigma_sq));
% err1 = immse(x_sample_1,x_reconstruct_1)
%err1 = immse(x_sample_1(:,15:18),x_reconstruct_1)

% err1 = immse(x_sample(:,1:4),x_reconstruct_1(:,1:4));
% err2 = immse(x_sample(:,5:8),x_reconstruct_1(:,5:8));
% err3 = immse(x_sample(:,9:12),x_reconstruct_1(:,9:12));
% err4 = immse(x_sample(:,13),x_reconstruct_1(:,13));
% err5 = immse(x_sample(:,14),x_reconstruct_1(:,14));
% err6 = immse(x_sample(:,15:18),x_reconstruct_1(:,15:18));
% err_1 = [err1,err2,err3,err4,err5,err6]


%% clear
load(['../results/',name,'_test2.mat'])
x_sample_2 = x_sample;
x_reconstruct_2 = x_reconstruct;
%x_reconstruct_var_2 = sqrt(exp(x_reconstruct_log_sigma_sq));
% err2 = immse(x_sample_1,x_reconstruct_2)
% err2_3 = immse(x_sample_2(:,8:9),x_reconstruct_2(:,8:9))

% err1 = immse(x_sample(:,1:4),x_reconstruct_1(:,1:4));
% err2 = immse(x_sample(:,5:8),x_reconstruct_1(:,5:8));
% err3 = immse(x_sample(:,9:12),x_reconstruct_1(:,9:12));
% err4 = immse(x_sample(:,13),x_reconstruct_1(:,13));
% err5 = immse(x_sample(:,14),x_reconstruct_1(:,14));
% err6 = immse(x_sample(:,15:18),x_reconstruct_1(:,15:18));
% err_2 = [err1,err2,err3,err4,err5,err6]

%%

figure
hold on
for i=1:4
    subplot(4,1,i)
    plot(x_sample_1(:,i+20),'k','linewidth',5); hold on
    plot(x_reconstruct_1(:,i+20),'color',color(1,:),'linewidth',2,'marker','s'); hold on
    %p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
    %p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
    plot(x_reconstruct_2(:,i+20),'color',color(2,:),'linewidth',2,'marker','o'); hold on
    %p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
    %p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
%     ylabel(bp{i})
    ylim([-1.2,1.2])
    %p1.Color(4) = 0.5;
    %p2.Color(4) = 0.5;
    %p3.Color(4) = 0.5;
    %p4.Color(4) = 0.5;
    set(gca,'XTickLabel','','Fontsize',12);
end

% for i=1:size(x_sample,2)
%     figure
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
% %     ylabel(bp{i})
%     ylim([-1.2,1.2])
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     set(gca,'XTickLabel','','Fontsize',12);
% end


%%
% close all
% 
% bp = {'Shoulder pitch','Shoulder roll','Shoulder yaw','Elbow',...
%     'Wrist pronosup.','Wrist pitch','Wrist yaw'};
% 
% figure
% hold on
% for i=1:4
%     subplot(4,1,i)
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
% %     ylabel(bp{i})
%     ylim([-1.2,1.2])
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     set(gca,'XTickLabel','','Fontsize',12);
% end
% j=i;

% figure
% hold on
% for i=5:8
%     subplot(4,1,i-j)
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
% %     ylabel(bp{i})
%     ylim([-1.2,1.2])
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     set(gca,'XTickLabel','','Fontsize',12);
% end
% j=i;
% 
% 
% figure
% hold on
% for i=9:12
%     subplot(4,1,i-j)
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
% %     ylabel(bp{i})
%     ylim([-1.2,1.2])
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     set(gca,'XTickLabel','','Fontsize',12);
% end
% j=i;
% 
% 
% figure
% hold on
% for i=13:14
%     subplot(4,1,i-j)
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
% %     ylabel(bp{i})
%     ylim([-1.2,1.2])
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     set(gca,'XTickLabel','','Fontsize',12);
% end
% j=i;
% 
% 
% figure
% hold on
% for i=15:18
%     subplot(4,1,i-j)
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
% %     ylabel(bp{i})
%     ylim([-1.2,1.2])
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     set(gca,'XTickLabel','','Fontsize',12);
% end


%%
% 
% figure
% hold on
% for i=1:3
%     subplot(1,3,i)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Joint pos. 0,1,2')
% 
% 
% figure
% hold on
% for i=4:7
%     subplot(1,4,i-3)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Joint pos. 3,4,5,6')
% 
% 
% figure
% hold on
% for i=8:10
%     subplot(1,3,i-7)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Vis. pos.')
% 
% 
% figure
% hold on
% for i=11:13
%     subplot(1,3,i-10)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Vis. pos. prev')
% 
% 
% figure
% hold on
% for i=14:15
%     subplot(1,3,i-13)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Touch and Sound')
% 
% 
% figure
% hold on
% for i=16:18
%     subplot(1,3,i-15)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Motors 0,1,2')
% 
% 
% figure
% hold on
% for i=19:22
%     subplot(1,4,i-18)
%     plot(x_sample(:,i),'k','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)+var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_rec_mean(:,i)-var(:,i),':','linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Motors 3,4,5,6')
% 


%%
% 
% figure
% hold on
% for i=1:3
%     subplot(1,3,i)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Joint pos. 0,1,2')
% 
% 
% figure
% hold on
% for i=4:7
%     subplot(1,4,i-3)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Joint pos. 3,4,5,6')
% 
% 
% figure
% hold on
% for i=8:10
%     subplot(1,3,i-7)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Vis. pos.')
% 
% 
% figure
% hold on
% for i=11:13
%     subplot(1,3,i-10)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Vis. pos. prev')
% 
% 
% figure
% hold on
% for i=14:15
%     subplot(1,3,i-13)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Touch and Sound')
% 
% 
% figure
% hold on
% for i=16:18
%     subplot(1,3,i-15)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Motors 0,1,2')
% 
% 
% figure
% hold on
% for i=19:22
%     subplot(1,4,i-18)
%     plot(x_sample_1(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_sample_2(:,i),'k-','linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),':','color',color(1,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
%     plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); hold on
%     plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),':','color',color(2,:),'linewidth',1); xlim([0 size(x_sample(:,i),1)]); 
% end
% title('Motors 3,4,5,6')





%%

% j=1;
% figure
% for i=[15,18]
%     subplot(1,2,j)
%     plot(x_sample_1(:,i),'k','linewidth',5); hold on
%     plot(x_sample_2(:,i),'k','linewidth',5); hold on
%     plot(x_reconstruct_1(:,i),'color',color(1,:),'linewidth',2,'marker','s'); hold on
%     p1=plot(x_reconstruct_1(:,i)+x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); hold on
%     p2=plot(x_reconstruct_1(:,i)-x_reconstruct_var_1(:,i),'color',color(1,:),'linewidth',2); 
%     plot(x_reconstruct_2(:,i),'color',color(2,:),'linewidth',2,'marker','o'); hold on
%     p3=plot(x_reconstruct_2(:,i)+x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); hold on
%     p4=plot(x_reconstruct_2(:,i)-x_reconstruct_var_2(:,i),'color',color(2,:),'linewidth',2); 
%     p1.Color(4) = 0.5;
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
%     p4.Color(4) = 0.5;
%     xlim([1,80])
%     xticks(0:20:80)
%     xlabel('Time steps','Fontsize',12);
% %     ylabel(mc{i-15})
%     ylim([-2.2,1.2])
%     
%     
%     j=j+1;
% end


%%
% close all
% 
% load(['../results/',name,'_test5_5.mat'])
% x_sample_5_5 = x_sample;
% x_reconstruct_5_5 = x_reconstruct;
% x_reconstruct_var_5_5 = sqrt(exp(x_reconstruct_log_sigma_sq));
% err5_5 = immse(x_sample_5_5(8:10),x_reconstruct_5_5(8:10))
% 
% 
% % figure
% % hold on
% % plot(x_sample_5_5(:,5))
% % plot(x_reconstruct_5_5(:,5))
% % 
% % figure
% % hold on
% % plot(x_sample_5_5(:,6))
% % plot(x_reconstruct_5_5(:,6))
% % 
% % figure
% % hold on
% % plot(x_sample_5_5(:,7))
% % plot(x_reconstruct_5_5(:,7))
% % 
% % figure
% % hold on
% % plot(x_sample_5_5(:,8))
% % plot(x_reconstruct_5_5(:,8))
% 
% figure
% hold on
% plot(x_sample_5_5(:,5),x_sample_5_5(:,6))
% plot(x_reconstruct_5_5(:,5),x_reconstruct_5_5(:,6))
% 
% figure
% hold on
% plot(x_sample_5_5(:,7),x_sample_5_5(:,8))
% plot(x_reconstruct_5_5(:,7),x_reconstruct_5_5(:,8))
% 
% %
% figure
% hold on
% plot(x_sample_5_5(:,15:18))
% plot(x_reconstruct_5_5(:,15:18))
% plot(x_reconstruct_5_5(:,15:18)+x_reconstruct_var_5_5(:,15:18),':','color',color(1,:),'linewidth',1); %xlim([0 size(x_sample(:,10),1)]); hold on
% plot(x_reconstruct_5_5(:,15:18)-x_reconstruct_var_5_5(:,15:18),':','color',color(1,:),'linewidth',1); %xlim([0 size(x_sample(:,10),1)]); 
% 
% 
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%
% close all
% clear
% 
% 
% name = 'mvae' %'mvae4j4vnosampling' %
% 
% color = lines(2);
% 
% 
% 
% load(['../results/',name,'_test6.mat'])
% x_sample_6 = x_sample;
% x_reconstruct_6 = x_reconstruct;
% x_reconstruct_var_6 = sqrt(exp(x_reconstruct_log_sigma_sq));
% err6 = immse(x_sample_6(8:10),x_reconstruct_6(8:10))
% 
% 
% % figure
% % hold on
% % plot(x_sample_6(:,5))
% % plot(x_reconstruct_6(:,5))
% % 
% % figure
% % hold on
% % plot(x_sample_6(:,6))
% % plot(x_reconstruct_6(:,6))
% % 
% % figure
% % hold on
% % plot(x_sample_6(:,7))
% % plot(x_reconstruct_6(:,7))
% % 
% % figure
% % hold on
% % plot(x_sample_6(:,8))
% % plot(x_reconstruct_6(:,8))
% % 
% figure
% hold on
% plot(x_sample_6(:,5),x_sample_6(:,6))
% plot(x_reconstruct_6(:,5),x_reconstruct_6(:,6))
% 
% % figure
% % hold on
% % plot(x_sample_6(:,7),x_sample_6(:,8))
% % plot(x_reconstruct_6(:,7),x_reconstruct_6(:,8))
% 
% %
% figure
% hold on
% plot(x_sample_6(:,15:18))
% plot(x_reconstruct_6(:,15:18))
% plot(x_reconstruct_6(:,15:18)+x_reconstruct_var_6(:,15:18),':','color',color(1,:),'linewidth',1); %xlim([0 size(x_sample(:,10),1)]); hold on
% plot(x_reconstruct_6(:,15:18)-x_reconstruct_var_6(:,15:18),':','color',color(1,:),'linewidth',1); %xlim([0 size(x_sample(:,10),1)]); 
% 
% 
% 
% %%
% % 
% % load(['../results/',name,'_test4.mat'])
% % x_sample_4 = x_sample;
% % x_reconstruct_4 = x_reconstruct;
% % x_reconstruct_var_4 = sqrt(exp(x_reconstruct_log_sigma_sq));
% % err4 = immse(x_sample_4(8:10),x_reconstruct_4(8:10))
% % 
% % 
% % % figure
% % % hold on
% % % plot(x_sample_4(:,5))
% % % plot(x_reconstruct_4(:,5))
% % % 
% % % figure
% % % hold on
% % % plot(x_sample_4(:,6))
% % % plot(x_reconstruct_4(:,6))
% % % 
% % % figure
% % % hold on
% % % plot(x_sample_4(:,7))
% % % plot(x_reconstruct_4(:,7))
% % % 
% % % figure
% % % hold on
% % % plot(x_sample_4(:,8))
% % % plot(x_reconstruct_4(:,8))
% % 
% % figure
% % hold on
% % plot(x_sample_4(:,5),x_sample_4(:,6))
% % plot(x_reconstruct_4(:,5),x_reconstruct_4(:,6))
% % 
% % figure
% % hold on
% % plot(x_sample_4(:,7),x_sample_4(:,8))
% % plot(x_reconstruct_4(:,7),x_reconstruct_4(:,8))
% % 
% % %
% % figure
% % hold on
% % plot(x_sample_4(:,15:18))
% % plot(x_reconstruct_4(:,15:18))
% % plot(x_reconstruct_4(:,15:18)+x_reconstruct_var_4(:,15:18),':','color',color(1,:),'linewidth',1); %xlim([0 size(x_sample(:,10),1)]); hold on
% % plot(x_reconstruct_4(:,15:18)-x_reconstruct_var_4(:,15:18),':','color',color(1,:),'linewidth',1); %xlim([0 size(x_sample(:,10),1)]); 
% 
