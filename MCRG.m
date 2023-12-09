%% clear
clear;close all;clc
%% Input Parameters (Known: y_t = 1, y_h = 1.875 exactly)
L = 20; %linear dimension of square lattice
Tc=(2/log(1+sqrt(2)));
Kc = 1/Tc; %Critical temperature Kc assumed to be known
K = Kc; h = 0; %Start on Critical manifold
nwarm = 1000; %number of warm up Monte Carlo sweeps
nmeas = 1000; %number of measurement Monte Carlo sweeps
interval = 10; %Take data every MC-steps/site interval
%RG analysis setting
Nc_even = 1; %number of coupling constants included in calculating T
Nc_odd = 1;
b = 2; %scaling factor
[Spin]=generate(L);
RunMCRG(Spin,L,K,h,nwarm,nmeas,interval,Nc_even,Nc_odd,b)
%% RunMCRG
function RunMCRG(Spin,L,K,h,nwarm,nmeas,interval,Nc_even,Nc_odd,b)
fprintf('running MCRG for lattice size = %i\n',L);
fprintf('setting K=%f\t',K); fprintf('h=%f\n',h);
%measurement accumulators for y_t
evenK = zeros(1,7);
evenK_1 = zeros(1,7);
mix_11 = zeros(7,7);
mix_01 = zeros(7,7);
%measurement accumulators for y_h
oddK = zeros(1,4);
oddK_1 = zeros(1,4);
mix_11_odd = zeros(4,4);
mix_01_odd = zeros(4,4);
%prepare simulation
energy=zeros(nmeas,1);
%warm system
for i = 1:nwarm
    [Spin]=metropolis(Spin,K,h);
end
%measure system
for i = 1:nmeas
    for j=1:interval
        [Spin]=metropolis(Spin,K,h);
    end
    %take measurements every (interval) steps
    [energy(i)]=Energy(Spin,K,h);
    Spin1 = RGTransform(Spin,b);   
    evenK = evenK + AllEvenCoupling(Spin);
    evenK_1 = evenK_1 + AllEvenCoupling(Spin1);
    oddK = oddK + AllOddCoupling(Spin);
    oddK_1 = oddK_1 + AllOddCoupling(Spin1);
    %A*B = C, B is unknown, A is symmetric
    mix_11 =mix_11 + reshape(kron(AllEvenCoupling(Spin1),AllEvenCoupling(Spin1)),7,7)';
    mix_01 =mix_01 + reshape(kron(AllEvenCoupling(Spin1),AllEvenCoupling(Spin)),7,7)';
    mix_11_odd =mix_11_odd + reshape(kron(AllOddCoupling(Spin1),AllOddCoupling(Spin1)),4,4)';
    mix_01_odd =mix_01_odd + reshape(kron(AllOddCoupling(Spin1),AllOddCoupling(Spin)),4,4)';
end
%Results
evenK = evenK / nmeas;
evenK_1 =evenK_1 / nmeas;
oddK = oddK  / nmeas;
oddK_1 = oddK_1 / nmeas;
mix_11 =mix_11 / nmeas;
mix_01 =mix_01 / nmeas;
mix_11_odd =mix_11_odd / nmeas;
mix_01_odd =mix_01_odd / nmeas;
%calculate y_t
MatA = mix_11 - reshape(kron(evenK_1,evenK_1),7,7)';
MatC = mix_01 - reshape(kron(evenK_1,evenK),7,7)';
LinRGMat = linsolve(MatA(1:Nc_even+1,1:Nc_even+1),MatC(1:Nc_even+1,1:Nc_even+1));
lmbd = eig(LinRGMat);
amplitude = abs(lmbd);
%Only the eigenvalue with maximum amplitude is important. This eigenvalue should generically be real
imax = max(amplitude);
y_t = log(imax)/log(b);
fprintf('exponent y_t = %f\n',y_t);
%calculate y_h
MatA = mix_11_odd - reshape(kron(oddK_1,oddK_1),4,4)';
MatC = mix_01_odd - reshape(kron(oddK_1,oddK),4,4)';
LinRGMat = linsolve(MatA(1:Nc_odd+1,1:Nc_odd+1),MatC(1:Nc_odd+1,1:Nc_odd+1));
lmbd = eig(LinRGMat);
amplitude = abs(lmbd);
%Only the eigenvalue with maximum amplitude is important. This eigenvalue should generically be real
imax = max(amplitude);
y_h = log(imax)/log(b);
fprintf('exponent y_h = %f\n',y_h);
end
%% neighbor
function [nbor]=neighbor(n)
%generate table of near neighbor
nbor=zeros(n*n,4);
for ispin=1:n*n
    iy= fix((ispin-1)/n)+1;
    ix=ispin-(iy-1)*n;
    ixp=ix+1-fix(ix/n)*n;
    iyp=iy+1-fix(iy/n)*n;
    ixm=ix-1+fix((n-ix+1)/n)*n;
    iym=iy-1+fix((n-iy+1)/n)*n;
    nbor(ispin,1)=(iy-1)*n+ixp;%右邻居
    nbor(ispin,2)=(iym-1)*n+ix;%上邻居
    nbor(ispin,3)=(iy-1)*n+ixm;%左邻居
    nbor(ispin,4)=(iyp-1)*n+ix;%下邻居
end
end
%% Energy
function [e]=Energy(Spin,K,h)
L=size(Spin,1);
[nbor]=neighbor(L);
reSpin=reshape(Spin,L*L,1);
e=0.0;
for i=1:L*L
    e=e + K * reSpin(i) * (reSpin(nbor(i,1)) + reSpin(nbor(i,2)));
end
e=e + h*sum(reSpin);
end
%% assignBlockSpin
function [blockSpin]=assignBlockSpin(totalSpin)
if totalSpin>0
    blockSpin=1;
elseif totalSpin<0
    blockSpin=-1;
else
    if rand>0.5
        blockSpin=1;
    else
        blockSpin=-1;
    end
end
end
%% RGtransform
function [newSpin]=RGTransform(Spin,b)
newL=fix(size(Spin,1)/b);
newSpin=zeros(newL,newL);
for i=1:newL
    for j=1:newL
        block=Spin(i*b-1:i*b,j*b-1:j*b);
        total=sum(block);
        newSpin(i,j)=assignBlockSpin(total);
    end
end
end
%% AllEvenCoupling
function [val]=AllEvenCoupling(Spin)
val=zeros(1,7);
L=size(Spin,1);
[nbor]=neighbor(L);
reSpin=reshape(Spin',L*L,1);
for j=1:L*L
    val(1)=val(1) + reSpin(j)*(reSpin(nbor(j,1))+reSpin(nbor(j,2)));%nearest neighbor (1,0)
    
    val(2)=val(2) + reSpin(j)*(reSpin(nbor(nbor(j,1),2))+reSpin(nbor(nbor(j,3),2)));%next nearest neighbor (1,1)
    
    val(3)=val(3) + reSpin(j)*(reSpin(nbor(nbor(j,1),1))+reSpin(nbor(nbor(j,2),2)));%3rd nearest neighbor (2,0)
    
    val(4)=val(4) + reSpin(j)*(reSpin(nbor(nbor(nbor(j,1),2),2))+reSpin(nbor(nbor(nbor(j,1),1),2)));%4th nearest neighbor (2,1)
    val(4)=val(4) + reSpin(j)*(reSpin(nbor(nbor(nbor(j,3),2),2))+reSpin(nbor(nbor(nbor(j,3),3),2)));
    
    val(5)=val(5) + reSpin(j)*(reSpin(nbor(nbor(nbor(nbor(j,1),1),2),2))+reSpin(nbor(nbor(nbor(nbor(j,3),3),3),2)));%5th nearest neighbor (2,2)
    
    val(6)=val(6) + reSpin(j)*reSpin(nbor(j,1))*reSpin(nbor(j,2))*reSpin(nbor(nbor(j,1),2));%plaquette
    
    val(7)=val(7) + reSpin(nbor(j,1))*reSpin(nbor(j,2))*reSpin(nbor(j,3))*reSpin(nbor(j,4));%sublattice plaquette
end
end
%% AllOddCoupling
function [val]=AllOddCoupling(Spin)
val=zeros(1,4);
L=size(Spin,1);
[nbor]=neighbor(L);
reSpin=reshape(Spin',L*L,1);
for j=1:L*L
    val(1)=val(1) + 0;%magnetization
    
    val(2)=val(2) + reSpin(j)*reSpin(nbor(j,1))*reSpin(nbor(nbor(j,1),2));%3 spin plaquette
    val(2)=val(2) + reSpin(j)*reSpin(nbor(j,2))*reSpin(nbor(nbor(j,2),3));
    val(2)=val(2) + reSpin(j)*reSpin(nbor(j,3))*reSpin(nbor(nbor(j,3),4));
    val(2)=val(2) + reSpin(j)*reSpin(nbor(j,4))*reSpin(nbor(nbor(j,4),1));
    
    val(3)=val(3) + reSpin(j)*reSpin(nbor(j,1))*reSpin(nbor(nbor(nbor(j,1),1),2));%3 spin angle
    val(3)=val(3) + reSpin(j)*reSpin(nbor(j,2))*reSpin(nbor(nbor(nbor(j,2),2),3));
    val(3)=val(3) + reSpin(j)*reSpin(nbor(j,3))*reSpin(nbor(nbor(nbor(j,3),3),4));
    val(3)=val(3) + reSpin(j)*reSpin(nbor(j,4))*reSpin(nbor(nbor(nbor(j,4),4),1));
    
    val(4)=val(4) + reSpin(j)*reSpin(nbor(j,1))*reSpin(nbor(nbor(j,1),1));%3 spin row
    val(4)=val(4) + reSpin(j)*reSpin(nbor(j,2))*reSpin(nbor(nbor(j,2),2));
end
val(1)=sum(reSpin);
end
%% metropolis update
function [Spin]=metropolis(Spin,K,h)
L=size(Spin,1);
reSpin=reshape(Spin',L*L,1);
[nbor]=neighbor(L);
for j =1:L*L
    cen=unidrnd(L*L); right=nbor(cen,1); up=nbor(cen,2); left=nbor(cen,3); down=nbor(cen,4);
    dh = 2 * h * reSpin(cen);
    dE = 2 * K * reSpin(cen) * (reSpin(right) + reSpin(left) + reSpin(up) + reSpin(down));
    if rand<exp(-1*(dE+dh))
        reSpin(cen)=-reSpin(cen);
    end
end
Spin=reshape(reSpin',L,L);
end
%% generate Spin
function [Spin]=generate(L)
Spin=ones(L,L);
for i=1:L
    for j=1:L
        if rand>0.5
            Spin(i,j)=-1;
        end
    end
end
end