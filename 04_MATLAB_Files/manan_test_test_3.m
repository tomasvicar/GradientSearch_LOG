clc;clear all;close all;


I=im2double(imread('qpi1.tif'));

sigma_x=10;
sigma_y=10;
theta=0;
[a,b,c]=sigma2abc(sigma_x,sigma_y,theta);

K=glogkernel(a, b, c);

q=imfilter(I,K,'replicate');
q=imerode(q,strel('disk',10));%distance filter
bw=imregionalmin(q);
bw=bw.*(q<-0.01);

s = regionprops(bw>0,'centroid');
centroids = round(cat(1, s.Centroid));

cxx=centroids(:,1);
cyy=centroids(:,2);

cxx=cxx+6;
cyy=cyy+6;

sigma_xx=5*ones(1,length(cxx));
sigma_yy=10*ones(1,length(cxx));
thetat=2*pi*rand(1,length(cxx)).*ones(1,length(cxx));

[aa,bb,cc]=sigma2abc(sigma_xx,sigma_yy,thetat);



at=zeros(100*100*100,1);
bt=zeros(100*100*100,1);
ct=zeros(100*100*100,1);
k=0;
for sx=linspace(7,20,100)
    for sy=linspace(7,20,100)
        for t=linspace(0,2*pi,100)
            k=k+1;
            [at(k),bt(k),ct(k)]=sigma2abc(sx,sy,t);
        end
    end
end

al=min(at);
bl=-0.0008;
cl=min(ct);


au=max(at);
bu=0.0008;
cu=max(ct);

% [at,bt,ct]=sigma2abc(20,7,0);
% 
% [att,btt,ctt]=sigma2abc(20,7,pi/2);


figure()
plot_gauss(I,cxx,cyy,aa,bb,cc)
figure()

N=20;

for k=1:2000
    
    

    for kk=1:length(cxx)
        cx=cxx(kk);
        cy=cyy(kk);
        a=aa(kk);
        b=bb(kk);
        c=cc(kk);

    
        fun = @(x)fun2min(x,I,cy,cx);

        x0=[a,b,c];
        
        
        
        
        
        ub = [au,bu,cu];
        

        lb = [al,bl,cl];
        

        A = [];
        b = [];
        Aeq = [];
        beq = [];
        nonlcon = [];
        options = optimoptions('fmincon','MaxFunctionEvaluations',500);

        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        a=x(1);
        b=x(2);
        c=x(3);
        
        
        cxx(kk)=cx;
        cyy(kk)=cy;
        aa(kk)=a;
        bb(kk)=b;
        cc(kk)=c;
        
    end
    
    plot_gauss(I,cxx,cyy,aa,bb,cc)
    title([num2str(k) 'par'])
    drawnow;
    
    
     for kk=1:length(cxx)
        cx=cxx(kk);
        cy=cyy(kk);
        a=aa(kk);
        b=bb(kk);
        c=cc(kk);
        
        
        v1=cy-N:cy+N;
        v2=cx-N:cx+N;
        data=I;
        if sum(v1<=0)>0
            v1=v1-min(v1)+1;
        end
        if sum(v2<=0)>0
            v2=v2-min(v2)+1;
        end
        if sum(v1>size(data,1))>0
            v1=v1-(max(v1)-size(data,1));
        end
        if sum(v2>size(data,2))>0
            v2=v2-(max(v2)-size(data,2));
        end
        q=imfilter(I(v1,v2),K,'replicate');

        [cyp,cxp]=find(q==min(q(:)),1,'first');
        cx=cx+cxp-N;
        cy=cy+cyp-N;

        
        cxx(kk)=cx;
        cyy(kk)=cy;
        aa(kk)=a;
        bb(kk)=b;
        cc(kk)=c;
     end
     
         plot_gauss(I,cxx,cyy,aa,bb,cc)
     title([num2str(k) 'x'])
     drawnow;
    
    
     
     

end


function plot_gauss(im,cxx,cyy,aa,bb,cc)
    
    hold off;
    imshow(im,'InitialMag', 'fit')
    hold on


    data=im;
    
    rng(1)
    for kk=1:length(cxx)
        
        
        aaqq=zeros(size(im));
        
        y=cxx(kk);
        x=cyy(kk);
        a=aa(kk);
        b=bb(kk);
        c=cc(kk);
        
        [G]= gaussian(a, b, c);
        
        
        v1=x-floor(size(G,1)/2):x+floor(size(G,1)/2);
        v2=y-floor(size(G,2)/2):y+floor(size(G,2)/2);

        if sum(v1<=0)>0
            v1=v1-min(v1)+1;
        end
        if sum(v2<=0)>0
            v2=v2-min(v2)+1;
        end
        if sum(v1>size(data,1))>0
            v1=v1-(max(v1)-size(data,1));
        end
        if sum(v2>size(data,2))>0
            v2=v2-(max(v2)-size(data,2));
        end
        
        aaqq(v1,v2)=aaqq(v1,v2)+G;
        
        qq=ones(size(aaqq));
        c=rand(3,1);
        img=cat(3,qq*c(1),qq*c(2),qq*c(3));
        
        h = imshow(img);
        set(h, 'AlphaData', aaqq) 
        
    end
    
    drawnow;
   

    
    
end


function im=point2im(cx,cy,imsize) 

im=zeros(imsize);

for kp=1:length(cx)
    a=cy(kp);
    b=cx(kp);
    a(a<1)=1;
    b(b<1)=1;
    im(a,b)=1;
end
im=im(1:imsize(1),1:imsize(2));



end




% function [LoG]= glogkernel(sigma_x, sigma_y, theta)
function [LoG]= glogkernel(a, b, c)



N =  100;




[X, Y] =  meshgrid( linspace(0, N, N + 1) - N/2, linspace(0, N, N + 1) - N / 2);


D2Gxx = ((2*a*X + 2*b*Y).^2 - 2*a) .*  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));
D2Gyy = ((2*b*X + 2*c*Y).^2 - 2*c) .*  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));

LoG = (D2Gxx + D2Gyy);
% Gaussian =  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));
% LoG = (D2Gxx + D2Gyy) ./  sum(Gaussian(:));


% LoG=(1/a)*(1/c)*LoG;

end



function [G]= gaussian(a, b, c)



N =  100;

[X, Y] =  meshgrid( linspace(0, N, N + 1) - N/2, linspace(0, N, N + 1) - N / 2);

G =  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));




end





function q=fun2min(par,data,x,y)
    LoG= glogkernel(par(1), par(2), par(3));
    
    v1=x-floor(size(LoG,1)/2):x+floor(size(LoG,1)/2);
    v2=y-floor(size(LoG,2)/2):y+floor(size(LoG,2)/2);
    
    if sum(v1<=0)>0
        v1=v1-min(v1)+1;
    end
    if sum(v2<=0)>0
        v2=v2-min(v2)+1;
    end
    if sum(v1>size(data,1))>0
        v1=v1-(max(v1)-size(data,1));
    end
    if sum(v2>size(data,2))>0
        v2=v2-(max(v2)-size(data,2));
    end
    
    
    w=data(v1,v2);
    
    q=sum(sum(w.*LoG));


end


function [a,b,c]=sigma2abc(sigma_x,sigma_y,theta)

a =  cos(theta) .^ 2 ./ (2 .* sigma_x .^ 2) +  sin(theta) .^ 2 ./ (2 .* sigma_y .^ 2);
b = - sin(2 .* theta) ./ (4 .* sigma_x .^ 2) + sin(2 .* theta) ./ (4 .* sigma_y .^ 2);
c =  sin(theta) .^ 2 ./ (2 .* sigma_x .^ 2) + cos(theta) .^ 2 ./ (2 .* sigma_y .^2);


end

