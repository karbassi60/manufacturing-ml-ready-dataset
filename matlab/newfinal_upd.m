%% OMEGA-D-26-00675.R1 | Reproducible MATLAB figures 2--8
% Requires MATLAB R2022b+ with Deep Learning Toolbox and Statistics and Machine Learning Toolbox.
% Data source: ml_ready_full 3(1).csv. All visualised performance is COMPUTED,
% never invented. This is a MATLAB reimplementation, not the Python run of the paper.
% Set fullRun=true for manuscript-scale repetition (computationally intensive).
clear; clc; close all; rng(42);
fullRun = true;
if fullRun
    seeds = 1:10; tuningSeeds = 1:3; maxEpoch = 400;
    learningRates = [1e-4 10^(-3.5) 1e-3 10^(-2.5) 1e-2 10^(-1.5) 1e-1 3e-1];
    l2Grid = [1e-5 1e-4 1e-3 1e-2 3e-2 1e-1 3e-1 1 3];
    fractions = .2:.2:1; noiseLevels = 0:.1:.5; sensitivitySeeds=1:3;
    nPerm=20;
else
    seeds=1:3; tuningSeeds=1; maxEpoch=75;
    learningRates=[1e-3 1e-2 1e-1 3e-1]; l2Grid=[1e-4 1e-3 1e-2 3e-2 1e-1];
    fractions=[.2 .6 1]; noiseLevels=[0 .25 .5]; sensitivitySeeds=1;
    nPerm=5;
end
batchSize=64; patience=30;
baseDir=fileparts(mfilename('fullpath'));
% Mac external drive location supplied by user. Spaces need no backslash
% when the path is stored in a quoted MATLAB character vector.
numbersFile = '/Volumes/Untitled 2/ml_ready_full.numbers';
externalCsv = '/Volumes/Untitled 2/ml_ready_full.csv';
% Try the external-drive CSV first; otherwise locate the supplied CSV.
% A .numbers file cannot be imported by MATLAB readtable: export it as CSV.
candidates = {externalCsv, ...
    fullfile(baseDir,'ml_ready_full 3(1).csv'), ...
    fullfile(baseDir,'ml_ready_full.csv'), ...
    fullfile(pwd,'ml_ready_full 3(1).csv'), ...
    fullfile(pwd,'ml_ready_full.csv'), ...
    fullfile(getenv('HOME'),'Downloads','ml_ready_full 3(1).csv'), ...
    fullfile(getenv('HOME'),'Downloads','ml_ready_full.csv')};
csvFile = '';
for k=1:numel(candidates)
    if isfile(candidates{k})
        csvFile = candidates{k};
        break;
    end
end
if isempty(csvFile)
    fprintf('Dataset not found automatically. Choose the provided CSV file.\n');
    [selectedName,selectedFolder] = uigetfile({'*.csv','CSV dataset (*.csv)'}, ...
        'Select ml_ready_full CSV (export .numbers from Apple Numbers first)');
    if ~isequal(selectedName,0)
        csvFile = fullfile(selectedFolder,selectedName);
    else
        error(['No CSV selected. The Apple Numbers file at %s cannot be read ' ...
            'directly by readtable. Export it from Numbers to CSV, or extract ' ...
            'the supplied package including its CSV into the same folder as this script.'],numbersFile);
    end
end
fprintf('Reading dataset: %s\n',csvFile);
outDir=fullfile(baseDir,'matlab_analysis_figures');
if ~exist(outDir,'dir'), mkdir(outDir); end
T=readtable(csvFile,'VariableNamingRule','preserve');
assert(height(T)==1000,'Expected 1,000 rows; verify dataset version.');
names=string(T.Properties.VariableNames);
required=["Production_Volume_Per_Hour","Department_Maintenance","Machine_Efficiency", ...
          "Workforce_Size","Raw_Materials_Cost","Shift_Hours"];
assert(all(ismember(required,names)),'Required original or transformed columns missing.');
% IMPORTANT: CSV does NOT contain Maintenance_Hours or Production_Line,
% although the paper specifies both; supplier and day/weekday encodings exist.
% The CSV does not preserve full dates: source ROW ORDER is treated as chronology.
missingOriginal=["Maintenance_Hours","Production_Line","Date","Total_Production"];
fprintf('Missing original variables: %s\n',strjoin(missingOriginal(~ismember(missingOriginal,names)),', '));
warning('Dataset differs from manuscript raw input; numerical results and 27-column PCA may not match the paper. No missing variable is fabricated.');
% Explicit leakage-control. Never use Machine_Efficiency, Production_Hours,
% Equipment_Age_Category_*, production volume, or derived proxy categories.
predNames=["Workforce_Size","Raw_Materials_Cost","Shift_Hours","Timestamp_Hour","Weekday", ...
           names(startsWith(names,'Supplier_'))];
predNames=unique(predNames,'stable');
% Weekday is already numeric-coded (do not claim 7 one-hot categories exist).
Xraw=double(T{:,cellstr(predNames)});
yOriginal=double(T.Production_Volume_Per_Hour);
yBin=double(T.Department_Maintenance);
% Efficiency regime cutpoints estimated from training observations ONLY.
idxTr=(1:550)'; idxVa=(551:700)'; idxCa=(701:800)'; idxTe=(801:1000)';
eff=double(T.Machine_Efficiency);
cut=quantile(eff(idxTr),[1/3 2/3]);
yOrd=1+double(eff>cut(1))+double(eff>cut(2));
% Training-only empirical rank-normal map; interpolate for held-out values.
[ys,~]=sort(yOriginal(idxTr));
u=((1:numel(ys))'-.5)/numel(ys);
[yu,~,g]=unique(ys); uUnique=accumarray(g,u,[],@mean);
yReg=norminv(min(1-1e-6,max(1e-6,interp1(yu,uUnique,yOriginal,'linear','extrap'))));
% Training-only feature normalization (zero variance columns harmless).
mu=mean(Xraw(idxTr,:),1); sigma=std(Xraw(idxTr,:),0,1); sigma(sigma<1e-9)=1;
X=(Xraw-mu)./sigma; X(~isfinite(X))=0;
D=struct('X',X,'yr',yReg,'yb',yBin,'yo',yOrd, ...
         'tr',idxTr,'va',idxVa,'ca',idxCa,'te',idxTe);
fprintf('Input: %d observations, %d available leakage-controlled features (paper specifies 27).\n',size(X,1),size(X,2));

%% Figure 2: PCA; target association; original-value heavy tails
[coeff,score,~,~,explained]=pca(X(idxTr,:));
allScores=(X-mean(X(idxTr,:),1))*coeff(:,1:3);
f=figure('Color','w','Position',[80 80 1440 820]);
subplot(2,3,1); scatter(allScores(:,1),allScores(:,2),14,yReg,'filled'); title('A. PCA: production (normal scores)'); xlabel('PC1');ylabel('PC2');colorbar;
subplot(2,3,2); scatter(allScores(:,1),allScores(:,2),14,yBin,'filled'); title('B. PCA: maintenance department');xlabel('PC1');ylabel('PC2');colorbar;
subplot(2,3,3); scatter(allScores(:,1),allScores(:,2),14,yOrd,'filled'); title('C. PCA: efficiency regime');xlabel('PC1');ylabel('PC2');colorbar;
subplot(2,3,4);
extraCols=["Machine_Efficiency","Production_Hours"];
assocNames=[predNames extraCols(ismember(extraCols,names))];
assocs=zeros(numel(assocNames),3);
for j=1:numel(assocNames)
    v=double(T{:,char(assocNames(j))});
    assocs(j,:)=[corr(v,yReg,'Rows','complete'),corr(v,yBin,'Rows','complete'),corr(v,yOrd,'Rows','complete')];
end
imagesc(assocs); caxis([-1 1]);colorbar;xticks(1:3);xticklabels({'Normal score','Department','Efficiency'});
yticks(1:numel(assocNames));yticklabels(cellstr(assocNames));set(gca,'TickLabelInterpreter','none','FontSize',8);
title('D. Pearson associations (*excluded)');
subplot(2,3,5); histogram(yOriginal,60);set(gca,'YScale','log');xlabel('Production volume per hour (raw)');ylabel('Count (log)');title('E. Heavy-tailed raw target');
subplot(2,3,6); scatter(T.Production_Hours,yOriginal,16,'filled');xlabel('Production Hours (excluded)');ylabel('Production volume per hour');title('F. Extreme ratios near -1');xline(-1,'--');
sgtitle(sprintf('Figure 2 | Exploratory analysis (PC1 %.1f%%; PC2 %.1f%%)',explained(1),explained(2)));
saveFig(f,outDir,'Figure_2_Data_exploration');

%% Figure 4: measured hyperparameter landscape (validation joint losses)
nL=numel(learningRates); nR=numel(l2Grid); landscape=nan(nR,nL);
fprintf('Computing validation-loss surface: %d parameter cells...\n',nL*nR);
for ir=1:nR
 for il=1:nL
    v=zeros(1,numel(tuningSeeds));
    for s=1:numel(tuningSeeds)
        try
            model=trainMTL(D,D.tr,learningRates(il),l2Grid(ir),tuningSeeds(s),maxEpoch,batchSize,patience);
            v(s)=model.bestVal;
        catch ME
            warning('Grid point LR=%g, L2=%g, seed=%d failed: %s',learningRates(il),l2Grid(ir),tuningSeeds(s),ME.message);
            v(s)=Inf;
        end
    end
    landscape(ir,il)=mean(v);
 end
end
assert(any(isfinite(landscape(:))),'All tuning runs failed; inspect data, toolboxes, and learning-rate grid.');
[~,ii]=min(landscape(:)); [iR,iL]=ind2sub(size(landscape),ii);
bestLR=learningRates(iL);bestL2=l2Grid(iR);
fprintf('Selected grid point: LR %.5g, lambda %.5g; validation loss %.5g\n',bestLR,bestL2,landscape(ii));
f=figure('Color','w','Position',[80 80 1500 420]);
[L,R]=meshgrid(log10(learningRates),log10(l2Grid));
subplot(1,4,1);surf(L,R,landscape);xlabel('log10(LR)');ylabel('log10(lambda)');zlabel('Validation loss');title('A. Measured surface');
subplot(1,4,2);contourf(L,R,landscape,18);hold on;plot(L(iR,iL),R(iR,iL),'kp','MarkerSize',12,'MarkerFaceColor','w');xlabel('log10(LR)');ylabel('log10(lambda)');title('B. Measured contours');colorbar;
subplot(1,4,3);semilogx(learningRates,landscape(iR,:),'-o');xline(bestLR,'--');xlabel('Learning rate');ylabel('Validation loss');title('C. Selected lambda');
subplot(1,4,4);semilogx(l2Grid,landscape(:,iL),'-o');xline(bestL2,'--');xlabel('L2 strength');ylabel('Validation loss');title('D. Selected LR');
sgtitle('Figure 4 | Measured validation optimization landscape');saveFig(f,outDir,'Figure_4_Optimization_landscape');

%% Fit same-seed joint and separate-task networks; preserve actual predictions
S=numel(seeds); nt=numel(idxTe);nv=numel(idxVa);
PRm=zeros(nt,S);PBm=zeros(nt,S);POm=zeros(nt,3,S);PRs=zeros(nt,S);PBs=zeros(nt,S);POs=zeros(nt,3,S);
PBval=zeros(nv,S);histExample=[]; bestExample=0;
metricM=zeros(S,3);metricS=zeros(S,3);
models=cell(S,1);
for k=1:S
 fprintf('Final refit %d/%d (seed %d)\n',k,S,seeds(k));
 m=trainMTL(D,D.tr,bestLR,bestL2,seeds(k),maxEpoch,batchSize,patience);
 models{k}=m;
 if k==1, histExample=m.history;bestExample=m.bestEpoch; end
 [PRm(:,k),PBm(:,k),POm(:,:,k)]=predictMTL(m,D.X(idxTe,:));
 [~,PBval(:,k),~]=predictMTL(m,D.X(idxVa,:));
 % Single-task ablations use equal hidden capacity and same training split.
 sm1=trainMTL(D,D.tr,bestLR,bestL2,seeds(k),maxEpoch,batchSize,patience,[1 0 0]);
 sm2=trainMTL(D,D.tr,bestLR,bestL2,seeds(k),maxEpoch,batchSize,patience,[0 1 0]);
 sm3=trainMTL(D,D.tr,bestLR,bestL2,seeds(k),maxEpoch,batchSize,patience,[0 0 1]);
 [PRs(:,k),~,~]=predictMTL(sm1,D.X(idxTe,:));
 [~,PBs(:,k),~]=predictMTL(sm2,D.X(idxTe,:));
 [~,~,POs(:,:,k)]=predictMTL(sm3,D.X(idxTe,:));
 metricM(k,:)=[r2(D.yr(idxTe),PRm(:,k)),prAuc(D.yb(idxTe),PBm(:,k)),macroF1(D.yo(idxTe),argmax3(POm(:,:,k)))];
 metricS(k,:)=[r2(D.yr(idxTe),PRs(:,k)),prAuc(D.yb(idxTe),PBs(:,k)),macroF1(D.yo(idxTe),argmax3(POs(:,:,k)))];
end
% Figure 5: seed averaged PR curve and probability reliability
pM=mean(PBm,2);pS=mean(PBs,2); yB=D.yb(idxTe);
[recM,preM]=prCurve(yB,pM);[recS,preS]=prCurve(yB,pS);
f=figure('Color','w','Position',[100 100 1100 470]);
subplot(1,2,1);plot(recM,preM,'LineWidth',1.7);hold on;plot(recS,preS,'LineWidth',1.7);yline(mean(yB),'k--');xlabel('Recall');ylabel('Precision');legend('Multi-task','Single-task','No-skill','Location','best');title('A. Test precision-recall');grid on;
subplot(1,2,2);[pp,oo]=reliabilityBins(yB,pM,5);plot([0 1],[0 1],'k--');hold on;plot(pp,oo,'o-','LineWidth',1.7);xlabel('Mean predicted probability');ylabel('Observed fraction positive');xlim([0 1]);ylim([0 1]);title('B. Test reliability (equal count)');grid on;
sgtitle('Figure 5 | Binary department classification');saveFig(f,outDir,'Figure_5_Binary_PR_reliability');
% Figure 6: repeated-seed performance (primary metrics)
f=figure('Color','w','Position',[100 100 1320 430]);
metricNames={'Production R^2','Binary PR-AUC','Efficiency macro-F1'};
for j=1:3
 subplot(1,3,j);hold on;
 for k=1:S,plot([1 2],[metricM(k,j) metricS(k,j)],'-','Color',[.7 .7 .7]);end
 scatter(ones(S,1),metricM(:,j),25,'filled');scatter(2*ones(S,1),metricS(:,j),25,'filled');
 errorbar([1 2],[mean(metricM(:,j)) mean(metricS(:,j))],[std(metricM(:,j)) std(metricS(:,j))],'ks','LineWidth',1.5);
 xlim([.7 2.3]);xticks([1 2]);xticklabels({'MTL','STL'});ylabel(metricNames{j});grid on;title(sprintf('%s; n=%d',metricNames{j},S));
end
sgtitle('Figure 6 | Stability across random initializations');saveFig(f,outDir,'Figure_6_Seed_stability');
% Figure 7: real epoch-by-epoch validation losses
f=figure('Color','w','Position',[100 100 1050 430]);
subplot(1,2,1);plot(histExample(:,1:3)./histExample(1,1:3),'LineWidth',1.5);xline(bestExample,'k--');legend('Regression','Binary','Ordinal','Early stop','Location','best');xlabel('Epoch');ylabel('Loss / epoch 1 loss');title('A. Relative task losses');grid on;
subplot(1,2,2);plot(sum(histExample(:,1:3),2),'LineWidth',1.5);xline(bestExample,'k--');xlabel('Epoch');ylabel('Unpenalized validation joint loss');title('B. Validation joint loss');grid on;
sgtitle('Figure 7 | Measured training dynamics');saveFig(f,outDir,'Figure_7_Training_dynamics');

%% Figure 3: grouped permutation drops on validation metrics, no test leakage
pR0=zeros(nv,S); pB0=PBval; pO0=zeros(nv,3,S);
for k=1:S,[pR0(:,k),~,pO0(:,:,k)]=predictMTL(models{k},D.X(idxVa,:));end
baseMetrics=[r2(D.yr(idxVa),mean(pR0,2)),prAuc(D.yb(idxVa),mean(pB0,2)), ...
 macroF1(D.yo(idxVa),argmax3(mean(pO0,3)))];
% Supplier one-hot columns must be permuted as a group.
groups={};groupNames={};
for j=1:numel(predNames)
 if startsWith(predNames(j),'Supplier_'),continue;end
 groups{end+1}=j;groupNames{end+1}=char(predNames(j)); %#ok<SAGROW>
end
groups{end+1}=find(startsWith(predNames,'Supplier_'));groupNames{end+1}='Supplier (joint)';
imp=zeros(numel(groups),3,nPerm);
for g=1:numel(groups)
 for rep=1:nPerm
    rng(5000+g*100+rep);Z=D.X(idxVa,:);p=randperm(nv);Z(:,groups{g})=Z(p,groups{g});
    a=zeros(nv,S);b=zeros(nv,S);c=zeros(nv,3,S);
    for k=1:S,[a(:,k),b(:,k),c(:,:,k)]=predictMTL(models{k},Z);end
    imp(g,:,rep)=baseMetrics-[r2(D.yr(idxVa),mean(a,2)),prAuc(D.yb(idxVa),mean(b,2)), ...
                          macroF1(D.yo(idxVa),argmax3(mean(c,3)))];
 end
end
f=figure('Color','w','Position',[100 100 1300 430]);
for t=1:3
 subplot(1,3,t);v=squeeze(imp(:,t,:));av=mean(v,2);ci=prctile(v,[2.5 97.5],2);
 errorbar(1:numel(groups),av,av-ci(:,1),ci(:,2)-av,'o');yline(0,'k--');xticks(1:numel(groups));xticklabels(groupNames);xtickangle(45);ylabel('Validation metric decrease');title(metricNames{t});grid on;
end
sgtitle('Figure 3 | Grouped validation permutation importance');saveFig(f,outDir,'Figure_3_Permutation_importance');

%% Figure 8: measured retraining with training fraction and input noise
A=nan(numel(noiseLevels),numel(fractions)); B=A;
for i=1:numel(fractions)
 for j=1:numel(noiseLevels)
  rr=zeros(numel(sensitivitySeeds),1);ff=rr;
  for k=1:numel(sensitivitySeeds)
   seed=sensitivitySeeds(k);rng(seed+1300);
   take=D.tr(randperm(numel(D.tr),max(30,round(fractions(i)*numel(D.tr)))));
   noisy=D; rng(seed+1800+j*100+i);
   noisy.X([D.tr;D.va;D.te],:)=D.X([D.tr;D.va;D.te],:) + noiseLevels(j)*randn(numel([D.tr;D.va;D.te]),size(D.X,2));
   m=trainMTL(noisy,take,bestLR,bestL2,seed,maxEpoch,batchSize,patience);
   [yr,~,yo]=predictMTL(m,noisy.X(D.te,:));
   rr(k)=r2(D.yr(D.te),yr);ff(k)=macroF1(D.yo(D.te),argmax3(yo));
  end
  A(j,i)=mean(rr);B(j,i)=mean(ff);
 end
end
[fX,fY]=meshgrid(fractions,noiseLevels);
f=figure('Color','w','Position',[100 100 1130 490]);
subplot(1,2,1);surf(fX,fY,A);xlabel('Training share');ylabel('Input noise SD');zlabel('Test R^2');title('A. Production');colorbar;
subplot(1,2,2);surf(fX,fY,B);xlabel('Training share');ylabel('Input noise SD');zlabel('Test macro-F1');title('B. Efficiency');colorbar;
sgtitle('Figure 8 | Empirical retraining sensitivity');saveFig(f,outDir,'Figure_8_Sensitivity');

%% Export analysis-ready metrics and provenance
writetable(array2table([seeds(:),metricM,metricS],'VariableNames', ...
 {'Seed','MTL_R2','MTL_PR_AUC','MTL_MacroF1','STL_R2','STL_PR_AUC','STL_MacroF1'}), ...
 fullfile(outDir,'metrics_by_seed.csv'));
writetable(array2table(landscape,'VariableNames',matlab.lang.makeValidName(compose('LR_%g',learningRates))), ...
 fullfile(outDir,'validation_loss_grid.csv'));
save(fullfile(outDir,'analysis_results.mat'),'metricM','metricS','landscape','learningRates','l2Grid', ...
 'fractions','noiseLevels','A','B','predNames','PRm','PBm','POm','PRs','PBs','POs','cut','idxTr','idxVa','idxCa','idxTe');
fprintf('\nCOMPLETE. Figures 2--8 and metrics in %s\n',outDir);

%% -------- Local functions (place at the bottom of the script) --------
function model=trainMTL(D,trainIDs,lr,l2,seed,maxEpoch,batchSize,patience,weights)
 if nargin<9,weights=[1 1 1];end
 rng(seed);p=size(D.X,2);
 dims=[p 64 32];
 P.W1=dlarray(single(randn(dims(2),dims(1))*sqrt(2/dims(1)))); P.b1=dlarray(zeros(dims(2),1,'single'));
 P.W2=dlarray(single(randn(dims(3),dims(2))*sqrt(2/dims(2)))); P.b2=dlarray(zeros(dims(3),1,'single'));
 P.Wr=dlarray(single(randn(1,32)*sqrt(2/32)));P.br=dlarray(single(0));
 P.Wb=dlarray(single(randn(1,32)*sqrt(2/32)));P.bb=dlarray(single(0));
 P.Wo=dlarray(single(randn(1,32)*sqrt(2/32)));P.bo=dlarray(single(0));
 P.t1=dlarray(single(-.5));P.delta=dlarray(single(.5));
 % All trainable tensors updated using Adam; early stop uses unpenalized validation loss.
 avg=[];sq=[];step=0;best=inf;bad=0;history=zeros(maxEpoch,3);
 Xv=single(D.X(D.va,:)'); rv=single(D.yr(D.va)');bv=single(D.yb(D.va)'); ov=single(D.yo(D.va)');
 for e=1:maxEpoch
   ids=trainIDs(randperm(numel(trainIDs)));
   for start=1:batchSize:numel(ids)
      ix=ids(start:min(start+batchSize-1,numel(ids)));
      xx=dlarray(single(D.X(ix,:)')); rr=dlarray(single(D.yr(ix)'));bb=dlarray(single(D.yb(ix)'));oo=dlarray(single(D.yo(ix)'));
      [~,grads]=dlfeval(@lossGrad,P,xx,rr,bb,oo,l2,weights);
      step=step+1;
      [P,avg,sq]=adamupdate(P,grads,avg,sq,step,lr);
   end
   [a,b,c]=evalLoss(P,Xv,rv,bv,ov);
   history(e,:)=[a b c];
   val=sum(weights.*history(e,:));
   if isfinite(val)&&val<best-1e-7,best=val;bestP=P;bestEpoch=e;bad=0;else,bad=bad+1;end
   if bad>=patience,break;end
 end
 if ~exist('bestP','var'),error('Optimization did not produce a finite validation loss; reduce learning rate.');end
 model.P=bestP;model.bestVal=best;model.bestEpoch=bestEpoch;model.history=history(1:e,:);
end
function [loss,grad]=lossGrad(P,X,yr,yb,yo,l2,weights)
 [pr,pb,c]=forward(P,X);
 r=mean((pr-yr).^2,'all');
 b=mean(log(1+exp(-abs(pb)))+max(pb,0)-yb.*pb,'all');
 % cumulative-link targets I(Y>1) and I(Y>2)
 c1=dlarray(single(extractdata(yo)>1));c2=dlarray(single(extractdata(yo)>2));
 o=mean(log(1+exp(-abs(c(1,:))))+max(c(1,:),0)-c1.*c(1,:) + ...
        log(1+exp(-abs(c(2,:))))+max(c(2,:),0)-c2.*c(2,:),'all');
 reg=0;
 for n={'W1','W2','Wr','Wb','Wo'},q=P.(n{1});reg=reg+sum(q.^2,'all');end
 loss=weights(1)*r+weights(2)*b+weights(3)*o+l2*reg;
 grad=dlgradient(loss,P);
end
function [r,b,c]=forward(P,X)
 z=max(P.W1*X+P.b1,0);z=max(P.W2*z+P.b2,0);
 r=P.Wr*z+P.br;b=P.Wb*z+P.bb;s=P.Wo*z+P.bo;
 t2=P.t1+log(1+exp(P.delta))+1e-4;
 c=[s-P.t1;s-t2];
end
function [r,b,o]=evalLoss(P,X,yr,yb,yo)
 [rp,bp,cp]=forward(P,dlarray(X));
 yp=gather(extractdata(rp)); bp=gather(extractdata(bp));cp=gather(extractdata(cp));
 r=mean((yp-yr).^2);b=mean(log(1+exp(-abs(bp)))+max(bp,0)-yb.*bp);
 t1=single(yo>1);t2=single(yo>2);
 o=mean(log(1+exp(-abs(cp(1,:))))+max(cp(1,:),0)-t1.*cp(1,:) + ...
        log(1+exp(-abs(cp(2,:))))+max(cp(2,:),0)-t2.*cp(2,:));
end
function [yr,yb,yo]=predictMTL(model,X)
 [a,b,c]=forward(model.P,dlarray(single(X')));
 yr=double(gather(extractdata(a(:))));
 yb=double(1./(1+exp(-gather(extractdata(b(:))))));
 cum=double(1./(1+exp(-gather(extractdata(c)))));
 yo=[1-cum(1,:);cum(1,:)-cum(2,:);cum(2,:)]';
 yo=max(0,yo);yo=yo./max(sum(yo,2),eps);
end
function x=argmax3(p),[~,x]=max(p,[],2);end
function out=r2(y,p),out=1-sum((y-p).^2)/sum((y-mean(y)).^2);end
function v=macroF1(y,p)
 v=zeros(3,1);for c=1:3,tp=sum(y==c & p==c);den=sum(y==c)+sum(p==c);if den>0,v(c)=2*tp/den;end,end
 v=mean(v);
end
function [rec,pre]=prCurve(y,s)
 [~,ord]=sort(s,'descend');y=y(ord);tp=cumsum(y==1);n=(1:numel(y))';
 rec=[0;tp/max(sum(y==1),1)];pre=[1;tp./n];
end
function a=prAuc(y,s)
 [r,p]=prCurve(y,s);a=sum(diff(r).*p(2:end));
end
function [means,positive]=reliabilityBins(y,s,n)
 [~,order]=sort(s);part=round(linspace(0,numel(y),n+1));means=zeros(n,1);positive=means;
 for i=1:n,ix=order(part(i)+1:part(i+1));means(i)=mean(s(ix));positive(i)=mean(y(ix));end
end
function saveFig(h,out,name)
 exportgraphics(h,fullfile(out,[name '.png']),'Resolution',220);
 try,exportgraphics(h,fullfile(out,[name '.pdf']),'ContentType','vector');catch,end
 close(h);
end
