clc;clear all;close all;


I=im2double(imread('qpi1.tif'));

sigma_x=10;
sigma_y=10;
theta=0;

K=glogkernel(sigma_x, sigma_y, theta);

q=imfilter(I,K,'replicate');
q=imerode(q,strel('disk',10));%distance filter
bw=imregionalmin(q);
bw=bw.*(q<-0.0005);

s = regionprops(bw>0,'centroid');
centroids = round(cat(1, s.Centroid));

cxx=centroids(:,1);
cyy=centroids(:,2);

cxx=cxx+6;
cyy=cyy+6;

sigma_xx=7*ones(1,length(cxx));
sigma_yy=7*ones(1,length(cxx));
thetat=0*ones(1,length(cxx));



figure()
plot_gauss(I,cxx,cyy,sigma_xx,sigma_yy,thetat)


N=20;

for k=1:2000
    
    

    for kk=1:length(cxx)
        cx=cxx(kk);
        cy=cyy(kk);
        sigma_x=sigma_xx(kk);
        sigma_y=sigma_yy(kk);
        theta=thetat(kk);

    
        fun = @(x)fun2min(x,I,cy,cx);

        x0=[sigma_x,sigma_y,theta];
        lb = [7,7,-10*pi];
        ub = [20,20,10*pi];

        A = [];
        b = [];
        Aeq = [];
        beq = [];
        nonlcon = [];
        options = optimoptions('fmincon','MaxFunctionEvaluations',500);

        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        sigma_x=x(1);
        sigma_y=x(2);
        theta=x(3);
        
        
        cxx(kk)=cx;
        cyy(kk)=cy;
        sigma_xx(kk)=sigma_x;
        sigma_yy(kk)=sigma_y;
        thetat(kk)=theta;
        
    end
    
    plot_gauss(I,cxx,cyy,sigma_xx,sigma_yy,thetat)
    title([num2str(k) 'par'])
    drawnow;
    
    
     for kk=1:length(cxx)
        cx=cxx(kk);
        cy=cyy(kk);
        sigma_x=sigma_xx(kk);
        sigma_y=sigma_yy(kk);
        theta=thetat(kk);
        
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
        sigma_xx(kk)=sigma_x;
        sigma_yy(kk)=sigma_y;
        thetat(kk)=theta;
     end
     
     plot_gauss(I,cxx,cyy,sigma_xx,sigma_yy,thetat)
     title([num2str(k) 'x'])
     drawnow;
    
    
     
     

end


function plot_gauss(im,cxx,cyy,sigma_xx,sigma_yy,thetat)

    aa=zeros(size(im));
    for kk=1:length(cxx)
        cx=cxx(kk);
        cy=cyy(kk);
        sigma_x=sigma_xx(kk);
        sigma_y=sigma_yy(kk);
        theta=thetat(kk);
        
        
        imm=point2im(cx,cy,size(im));
        
        point=zeros(101);
        point(51,51)=1;
        point=imgaussfilt(point,[sigma_x,sigma_y]);
        
        point=imrotate(point,theta);
        imm=conv2(imm,point,'same');
        
        imm=imm/max(imm(:));
        aa=max(cat(3,aa,imm),[],3);
    end
    
    a=zeros(size(im,1),size(im,2),3);
    a(:,:,1)=im;
    a(:,:,2)=im;
    a(:,:,3)=im;
    
    
    a(:,:,1)=a(:,:,1)+aa;
    
    imshow(a,[])
    
    
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



s=max(sigma_x,sigma_y);

N =  ceil(2 * 3 * s);

N=floor(N/2)*2;



[X, Y] =  meshgrid( linspace(0, N, N + 1) - N/2, linspace(0, N, N + 1) - N / 2);



D2Gxx = ((2*a*X + 2*b*Y).^2 - 2*a) .*  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));
D2Gyy = ((2*b*X + 2*c*Y).^2 - 2*c) .*  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));

Gaussian =  exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));
LoG = (D2Gxx + D2Gyy) ./  sum(Gaussian(:));






end


function q=fun2min(par,data,x,y)
    LoG= glogkernel(par(2), par(1), par(3));
    
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
    
    q=sum(sum(w.*(LoG*par(1)*par(2))));


end


function q=abc2sigma(a,b,c)




end


function [a,b,c]=sigma2abc(sigma_x,sigma_y,theta)

a =  cos(theta) ^ 2 / (2 * sigma_x ^ 2) +  sin(theta) ^ 2 / (2 * sigma_y ^ 2);
b = - sin(2 * theta) / (4 * sigma_x ^ 2) + sin(2 * theta) / (4 * sigma_y ^ 2);
c =  sin(theta) ^ 2 / (2 * sigma_x ^ 2) + cos(theta) ^ 2 / (2 * sigma_y ^ 2);


end

