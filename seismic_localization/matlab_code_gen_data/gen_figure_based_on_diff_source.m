sTrue=slownessMap3('sd');


figure(1);
clf                 %  edge gap   vertical    horizontal
% ha=tight_subplot(1,5,[0.03 0.02],[.22 .22],[.095 .05]);
ha=tight_subplot(1,5,[0.03 0.04],[.22 .22],[.1 .05]);
axes(ha(1))
a = ha(1);
pos = get(a,'Position');
imagesc(sTrue,[0.3 0.5]);colormap("jet")
h=colorbar('northoutside');
set(a,'Position',pos) % fixing image size from colorbar addition
xlabel(h,'(a) Slowness (s/km)')
% xlabel('(a)')
% ylabel('Range (km)')
set(gca,'Xtick',[1,20:20:100])
xticklabels({0,0.2,0.4,0.6,0.8,1})
set(gca,'Ytick',[1,20:20:100])
yticklabels({1,0.8,0.6,0.4,0.2,0})
axis square;
% title('(a)');
    

axes(ha(2));
b = ha(2);
pos = get(b,'Position');
hold on
% set(gca,'Ydir','reverse')

for m =1:length(X1)
    plot([X1(m,1),X2(m,1)],[X1(m,2),X2(m,2)],'k','color',[0,0,0]+0.6,'linewidth',1,'DisplayName','ray path');
end

recX = RecLoc(:,1);
recY = RecLoc(:,2);
plot(recX,recY,'o','markersize',8,'linewidth',2, 'DisplayName','sensor');

% xSta = RecLoc(:,1);
%　ySta = RecLoc(:,2);
plot(xSta,ySta,'rx','markersize',8,'linewidth',2,'DisplayName','source');
% xlabel('(b)')
hPlots = flip(findall(gcf,'Type','Line'));
legend(hPlots(401:402),'Location','northoutside','Orientation','horizontal')
set(b,'Position',pos)
grid on
grid minor
% hold off
set(gca,'Xtick',[1,20:20:100])
xticklabels({0,0.2,0.4,0.6,0.8,1})
set(gca,'Ytick',[1,20:20:100])
yticklabels({0,0.2,0.4,0.6,0.8,1})
axis square;

% title('(b)');
%%%to invert
NoSrc = 25;
%gen csv file, used in gen_ray_path_script_fix_Rec
GenCsv = false;
% user inverted slowness map
use_invert = false;
NFCT = 0.02;
gen_ray_path_script_fix_Rec; % code to configure sensor array, and to calculate travel times assuming straight-rays
conventional_tomo_2pct_noise;
axes(ha(3))
a = ha(3);
pos = get(a,'Position');
imagesc(sInv,[0.3 0.5]);colormap("jet")
h=colorbar('northoutside');
set(a,'Position',pos) % fixing image size from colorbar addition
xlabel(h,'(c) Slowness (s/km)')
% xlabel('(c)')
%ylabel('Range (km)')
set(gca,'Xtick',[1,20:20:100])
xticklabels({0,0.2,0.4,0.6,0.8,1})
set(gca,'Ytick',[1,20:20:100])
yticklabels({1,0.8,0.6,0.4,0.2,0})
axis square;
% title('(c)');

NoSrc = 50;
%gen csv file, used in gen_ray_path_script_fix_Rec
GenCsv = false;
% user inverted slowness map
use_invert = false;
gen_ray_path_script_fix_Rec;
conventional_tomo_wj_2pct_noise;
axes(ha(4))
a = ha(4);
pos = get(a,'Position');
imagesc(sInv,[0.3 0.5]);colormap("jet")
h=colorbar('northoutside');
set(a,'Position',pos) % fixing image size from colorbar addition
xlabel(h,'(d) Slowness (s/km)')
% xlabel('(d)')
%ylabel('Range (km)')
set(gca,'Xtick',[1,20:20:100])
xticklabels({0,0.2,0.4,0.6,0.8,1})
set(gca,'Ytick',[1,20:20:100])
yticklabels({1,0.8,0.6,0.4,0.2,0})
axis square;
% title('(d)');
NoSrc = 100;
%gen csv file, used in gen_ray_path_script_fix_Rec
GenCsv = false;
% user inverted slowness map
use_invert = false;
gen_ray_path_script_fix_Rec;
conventional_tomo_wj_2pct_noise;
axes(ha(5))
a = ha(5);
pos = get(a,'Position');
imagesc(sInv,[0.3 0.5]);colormap("jet")
h=colorbar('northoutside');
set(a,'Position',pos) % fixing image size from colorbar addition
xlabel(h,'(e) Slowness (s/km)')
% xlabel('(e)')
%ylabel('Range (km)')
set(gca,'Xtick',[1,20:20:100])
xticklabels({0,0.2,0.4,0.6,0.8,1})
set(gca,'Ytick',[1,20:20:100])
yticklabels({1,0.8,0.6,0.4,0.2,0})
axis square;
set(findall(gcf,'-property','FontSize'),'FontSize',14, 'FontName','Times New Roman');

NoSrc = 50;
%gen csv file, used in gen_ray_path_script_fix_Rec
GenCsv = false;
% user inverted slowness map
use_invert = false;
gen_ray_path_script_fix_Rec;
conventional_tomo_wj_2pct_noise;